from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class GainOptimizationMode(Enum):
    """Enumeration of gain optimization modes."""

    NONE = auto()  # No gain optimization
    INDIVIDUAL = auto()  # Optimize individual p_gain and d_gain for each actuator
    SIMPLE = auto()  # Optimize just trans/rot p_gains with d_gains as 2*sqrt(p_gain)


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
        gain_mode: GainOptimizationMode = GainOptimizationMode.NONE,
        gain_limits: Optional[Dict[str, float]] = None,
    ):
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            planning_horizon: The number of control steps (T) to plan over.
            sim_steps_per_control_step: The number of simulation steps to take
                                        for each control step.
            trace_sites: A list of site names to visualize with traces.
            gain_mode: The gain optimization mode to use.
            gain_limits: Optional dictionary with custom gain limits. Keys can include:
                         'p_min', 'p_max', 'd_min', 'd_max' for INDIVIDUAL mode
                         'trans_p_min', 'trans_p_max', 'rot_p_min', 'rot_p_max' for SIMPLE mode
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)
        self.planning_horizon = planning_horizon
        self.sim_steps_per_control_step = sim_steps_per_control_step
        self.gain_mode = gain_mode

        # Set actuator limits
        self.u_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.u_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )

        # Original control dimensions
        self.nu_ctrl = mj_model.nu

        # Default gain limits
        self._default_gain_limits = {
            # INDIVIDUAL mode defaults
            "p_min": 5.0,
            "p_max": 30.0,
            "d_min": 1.0,
            "d_max": 10.0,
            # SIMPLE mode defaults
            "trans_p_min": 5.0,
            "trans_p_max": 30.0,
            "rot_p_min": 5.0,
            "rot_p_max": 30.0,
        }

        # Override with any custom gain limits
        if gain_limits:
            self._default_gain_limits.update(gain_limits)

        # Configure based on gain optimization mode
        if gain_mode == GainOptimizationMode.INDIVIDUAL:
            # Total control dimensions including gains (nu + 2*nu)
            self.nu_total = mj_model.nu * 3

            # Set gain limits based on defaults or custom values
            self.p_gain_min = jnp.ones(mj_model.nu) * self._default_gain_limits["p_min"]
            self.p_gain_max = jnp.ones(mj_model.nu) * self._default_gain_limits["p_max"]
            self.d_gain_min = jnp.ones(mj_model.nu) * self._default_gain_limits["d_min"]
            self.d_gain_max = jnp.ones(mj_model.nu) * self._default_gain_limits["d_max"]

            # Extend control limits to include gains
            self.u_min = jnp.concatenate([self.u_min, self.p_gain_min, self.d_gain_min])
            self.u_max = jnp.concatenate([self.u_max, self.p_gain_max, self.d_gain_max])

        elif gain_mode == GainOptimizationMode.SIMPLE:
            # Total control dimensions including 2 p_gains (trans and rot)
            self.nu_total = mj_model.nu + 2

            # Default simple gain limits - can be overridden by subclasses
            self.trans_p_gain_min = self._default_gain_limits["trans_p_min"]
            self.trans_p_gain_max = self._default_gain_limits["trans_p_max"]
            self.rot_p_gain_min = self._default_gain_limits["rot_p_min"]
            self.rot_p_gain_max = self._default_gain_limits["rot_p_max"]

            # Extend control limits to include the two p_gains
            self.u_min = jnp.concatenate(
                [self.u_min, jnp.array([self.trans_p_gain_min, self.rot_p_gain_min])]
            )
            self.u_max = jnp.concatenate(
                [self.u_max, jnp.array([self.trans_p_gain_max, self.rot_p_gain_max])]
            )
            # Assert the size of the control vector is correct
            assert self.u_min.shape == self.u_max.shape == (mj_model.nu + 2,), (
                "Control vector size mismatch"
            )
        else:  # GainOptimizationMode.NONE
            self.nu_total = mj_model.nu

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
            control: The extended control vector based on the gain optimization mode

        Returns:
            Tuple of (control_actions, p_gains, d_gains)
        """
        if self.gain_mode == GainOptimizationMode.NONE:
            return control, None, None

        elif self.gain_mode == GainOptimizationMode.INDIVIDUAL:
            u = control[: self.nu_ctrl]
            p_gains = control[self.nu_ctrl : 2 * self.nu_ctrl]
            d_gains = control[2 * self.nu_ctrl : 3 * self.nu_ctrl]
            return u, p_gains, d_gains

        elif self.gain_mode == GainOptimizationMode.SIMPLE:
            # Assert that the control vector is of the correct length 6
            assert self.nu_ctrl == 6, (
                "Control vector length must be 6 for SIMPLE gain optimization"
            )

            u = control[: self.nu_ctrl]
            trans_p_gain = control[self.nu_ctrl + 1] * jnp.ones(self.nu_ctrl // 2)
            rot_p_gain = control[self.nu_ctrl + 2] * jnp.ones(self.nu_ctrl // 2)

            p_gains = jnp.concatenate([trans_p_gain, rot_p_gain])

            # Calculate d_gains as 2*sqrt(p_gain)
            d_gains = 2.0 * jnp.sqrt(p_gains)

            return u, p_gains, d_gains
