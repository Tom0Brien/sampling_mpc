from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class NugusHeight(Task):
    """NUGUS robot to maintain a target height for the torso."""

    def __init__(
        self,
        planning_horizon: int = 5,
        sim_steps_per_control_step: int = 5,
        target_height: float = 0.5,  # Default target height in meters
        optimize_gains: bool = False,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/nugus/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["imu"],
            optimize_gains=optimize_gains,
        )

        # Store specific body/site IDs for later reference
        self.torso_id = mj_model.body("torso").id
        self.imu_id = mj_model.site("imu").id
        self.target_height = target_height

        # Store the joint actuator limits
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

        # Define actuator gains if we're optimizing them
        if optimize_gains:
            self.p_gain_min = jnp.ones(mj_model.nu) * 15
            self.p_gain_max = jnp.ones(mj_model.nu) * 30.0
            self.d_gain_min = jnp.ones(mj_model.nu) * 5
            self.d_gain_max = jnp.ones(mj_model.nu) * 20
            self.u_min = jnp.concatenate([self.u_min, self.p_gain_min, self.d_gain_min])
            self.u_max = jnp.concatenate([self.u_max, self.p_gain_max, self.d_gain_max])

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages maintaining target height and minimizing control effort."""
        state_cost = self.terminal_cost(state)

        # Compute the control cost: sum of squared actuator forces
        control_cost = jnp.sum(jnp.square(state.actuator_force))

        # Compute the velocity cost to encourage smooth movements
        velocity_cost = jnp.sum(jnp.square(state.qvel))

        return state_cost + 0.01 * control_cost + 0.005 * velocity_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T) penalizes deviation from target height."""
        # Get the current torso height
        current_height = state.site_xpos[self.imu_id][2]  # z-coordinate

        # Calculate height error
        height_error = current_height - self.target_height
        height_cost = jnp.square(height_error)

        # We also want to encourage the robot to stay balanced (upright)
        # Use the IMU site's z-axis to determine if the robot is upright
        upvector = state.site_xmat[self.imu_id][
            2::3
        ]  # Get the z-column of the rotation matrix
        upright_cost = jnp.sum(jnp.square(upvector - jnp.array([0, 0, 1])))

        # # Compute foot contact forces to encourage stable stance
        # left_foot_pos = state.site_xpos[state.site("left_foot").id]
        # right_foot_pos = state.site_xpos[state.site("right_foot").id]
        # foot_height_diff = jnp.abs(left_foot_pos[2] - right_foot_pos[2])

        # # Add small penalty for foot height difference to encourage even stance
        # foot_cost = jnp.square(foot_height_diff)

        return 50.0 * height_cost + 20.0 * upright_cost  # + 10.0 * foot_cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains."""
        multiplier = jax.random.uniform(
            rng, self.model.actuator_gainprm[:, 0].shape, minval=0.9, maxval=1.1
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}
