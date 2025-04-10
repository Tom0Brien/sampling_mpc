from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class ControlMode(Enum):
    """Enumeration of control modes."""

    # Controls are based on mujoco model
    GENERAL = auto()
    # Cartesian impedance control, assumes model has torque actuators and single end-effector
    CARTESIAN = auto()


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        planning_horizon: int,
        sim_steps_per_control_step: int,
        trace_sites: Sequence[str] = [],
        control_mode: ControlMode = ControlMode.GENERAL,
    ):
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            planning_horizon: The number of control steps (T) to plan over.
            sim_steps_per_control_step: The number of simulation steps to take
                                        for each control step.
            trace_sites: A list of site names to visualize with traces.
            control_mode: The control mode to use.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)
        self.planning_horizon = planning_horizon
        self.sim_steps_per_control_step = sim_steps_per_control_step
        self.control_mode = control_mode

        # For cartesian control modes, set default end-effector site ID
        # This should be overridden by subclasses
        self.ee_site_id = 0

        # Cartesian control gains defaults
        self.Kp = jnp.diag(jnp.array([300.0, 300.0, 300.0, 50.0, 50.0, 50.0]))
        self.Kd = 2.0 * jnp.sqrt(self.Kp)
        self.nullspace_stiffness = 10.0
        self.q_d_nullspace = jnp.zeros(mj_model.nv)

        # Configure control dimensions and limits based on control mode
        if self.control_mode == ControlMode.GENERAL:
            self.nu_ctrl = self.mj_model.nu
            self.nu_total = self.mj_model.nu
            self.u_min = jnp.where(
                self.mj_model.actuator_ctrllimited,
                self.mj_model.actuator_ctrlrange[:, 0],
                -jnp.inf,
            )
            self.u_max = jnp.where(
                self.mj_model.actuator_ctrllimited,
                self.mj_model.actuator_ctrlrange[:, 1],
                jnp.inf,
            )
        elif self.control_mode == ControlMode.CARTESIAN:
            self.nu_ctrl = 6  # 3D position and orientation of end-effector
            self.nu_total = 6
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")

        # Timestep for each control step
        self.dt = mj_model.opt.timestep * sim_steps_per_control_step

        # Get site IDs for points we want to trace
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}

    def extract_gains(
        self, control: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Extract control and gain components from the extended control vector.

        Args:
            control: The extended control vector based on the control mode

        Returns:
            Tuple of (control_actions, p_gains, d_gains)
        """
        if (
            self.control_mode == ControlMode.GENERAL
            or self.control_mode == ControlMode.CARTESIAN
        ):
            return control, None, None
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
