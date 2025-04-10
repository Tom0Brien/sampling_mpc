from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Tuple, Dict

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.risk import AverageCost, RiskStrategy
from hydrax.task_base import Task, ControlMode
from hydrax.controllers.impedance_controllers import impedance_control_mjx


@dataclass
class Trajectory:
    """Data class for storing rollout data.

    Attributes:
        controls: Control actions for each time step (size T).
        costs: Costs associated with each time step (size T+1).
        trace_sites: Positions of trace sites at each time step (size T+1).
    """

    controls: jax.Array
    costs: jax.Array
    trace_sites: jax.Array

    def __len__(self):
        """Return the number of time steps in the trajectory (T)."""
        return self.costs.shape[-1] - 1


class SamplingBasedController(ABC):
    """Abstract base class for sampling-based MPC algorithms."""

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
    ):
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combine costs from different randomizations.
                           Defaults to AverageCost if None.
            seed: The random seed for domain randomization.
        """
        self.task = task
        self.num_randomizations = max(num_randomizations, 1)
        self.risk_strategy = risk_strategy or AverageCost()

        # Initialize with single model (no domain randomization) by default
        self.model = task.model
        self.randomized_axes = None

        # Setup domain randomization if enabled
        if self.num_randomizations > 1:
            self._setup_domain_randomization(seed)

    def _setup_domain_randomization(self, seed: int) -> None:
        """Set up domain randomized models.

        Args:
            seed: Random seed for generating domain randomizations.
        """
        rng = jax.random.key(seed)
        rng, subrng = jax.random.split(rng)
        subrngs = jax.random.split(subrng, self.num_randomizations)

        # Create randomized versions of the model
        randomizations = jax.vmap(self.task.domain_randomize_model)(subrngs)
        self.model = self.task.model.tree_replace(randomizations)

        # Keep track of which elements of the model have randomization
        self.randomized_axes = jax.tree.map(lambda x: None, self.task.model)
        self.randomized_axes = self.randomized_axes.tree_replace(
            {key: 0 for key in randomizations.keys()}
        )

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Sample and clip random control sequences
        controls, params = self.sample_controls(params)
        controls = jnp.clip(controls, self.task.u_min, self.task.u_max)

        # Roll out the control sequences with domain randomizations
        rng, dr_rng = jax.random.split(params.rng)
        rollouts = self.rollout_with_randomizations(state, controls, dr_rng)
        params = params.replace(rng=rng)

        # Update the policy parameters based on the combined costs
        params = self.update_params(params, rollouts)
        return params, rollouts

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        controls: jax.Array,
        rng: jax.Array,
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            controls: The control sequences, size (num rollouts, horizon - 1).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Initialize states for each rollout
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        # Apply domain randomization to initial states if enabled
        if self.num_randomizations > 1:
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(states, subrngs)
            states = states.tree_replace(randomizations)

        # Apply control sequences across all domain randomizations
        _, rollouts = jax.vmap(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None)
        )(self.model, states, controls)

        # Combine costs from different domain randomizations
        costs = self.risk_strategy.combine_costs(rollouts.costs)

        # Return combined trajectory
        return Trajectory(
            controls=rollouts.controls[0],  # identical over randomizations
            costs=costs,
            trace_sites=rollouts.trace_sites[0],  # visualization only
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def eval_rollouts(
        self, model: mjx.Model, state: mjx.Data, controls: jax.Array
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences in parallel and compute costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, size (num rollouts, horizon - 1).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = mjx.forward(model, x)  # compute site positions

            # Apply control based on control mode
            control = None
            if self.task.control_mode == ControlMode.GENERAL:
                # Standard control mode - direct control application
                control = u
            elif self.task.control_mode == ControlMode.CARTESIAN:
                # Cartesian impedance control
                control = impedance_control_mjx(
                    model_mjx=model,
                    data_mjx=x,
                    p_des=u[:3],
                    eul_des=u[3:],
                    Kp=self.task.Kp,
                    Kd=self.task.Kd,
                    nullspace_stiffness=self.task.nullspace_stiffness,
                    q_d_nullspace=self.task.q_d_nullspace,
                    site_id=self.task.ee_site_id,
                )
            else:
                # Unsupported control mode
                raise ValueError(f"Control mode {self.task.control_mode} not supported")

            cost = self.task.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)
            # Advance state
            x = jax.lax.fori_loop(
                0,
                self.task.sim_steps_per_control_step,
                lambda _, x: mjx.step(model, x),
                x.replace(ctrl=control),
            )

            return x, (x, cost, sites)

        # Run simulation for all timesteps
        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
        )

        # Add terminal cost and trace site
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, Trajectory(
            controls=controls,
            costs=costs,
            trace_sites=trace_sites,
        )

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def init_params(self, initial_control: jax.Array = None) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            initial_control: Optional initial control to initialize parameters.

        Returns:
            The initial policy parameters.
        """
        pass

    @abstractmethod
    def sample_controls(self, params: Any) -> Tuple[jax.Array, Any]:
        """Sample a set of control sequences U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g., mean, std).

        Returns:
            A control sequences U, size (num rollouts, horizon - 1).
            Updated parameters (e.g., with a new PRNG key).
        """
        pass

    @abstractmethod
    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        """Update the policy parameters π(params) using the rollouts.

        Args:
            params: The current policy parameters.
            rollouts: The rollouts obtained from the current policy.

        Returns:
            The updated policy parameters.
        """
        pass

    @abstractmethod
    def get_action(self, params: Any, t: float) -> jax.Array:
        """Get the control action at a given point along the trajectory.

        Args:
            params: The policy parameters, U ~ π(params).
            t: The time (in seconds) from the start of the trajectory.

        Returns:
            The control action u(t).
        """
        pass
