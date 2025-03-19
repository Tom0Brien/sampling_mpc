from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.util import mat_to_quat, eul_to_quat, orientation_error


class FrankaReach(Task):
    """Franka to reach a target position."""

    def __init__(
        self,
        planning_horizon: int = 5,
        sim_steps_per_control_step: int = 5,
        optimize_gains: bool = False,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_scene_reach.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["gripper"],
            optimize_gains=optimize_gains,
        )

        self.gripper_id = mj_model.site("gripper").id
        self.reference_id = mj_model.site("reference").id

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
        if optimize_gains:
            self.p_gain_min = jnp.ones(mj_model.nu) * 15
            self.p_gain_max = jnp.ones(mj_model.nu) * 30.0
            self.d_gain_min = jnp.ones(mj_model.nu) * 5
            self.d_gain_max = jnp.ones(mj_model.nu) * 20
            self.u_min = jnp.concatenate([self.u_min, self.p_gain_min, self.d_gain_min])
            self.u_max = jnp.concatenate([self.u_max, self.p_gain_max, self.d_gain_max])

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        state_cost = self.terminal_cost(state)
        # Compute the control cost: sum of squared actuator forces
        control_cost = jnp.sum(jnp.square(state.actuator_force))
        return state_cost + 0.01 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # Use mocap position as the desired pose
        desired_position = state.mocap_pos[0]
        desired_orientation = state.mocap_quat[0]

        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.gripper_id] - desired_position)
        )
        # Quaternion difference - compute angular distance between quaternions
        current_rot = state.site_xmat[self.gripper_id].reshape((3, 3))
        current_quat = mat_to_quat(current_rot)
        ori_error = orientation_error(current_quat, desired_orientation, current_rot)
        orientation_cost = jnp.sum(jnp.square(ori_error))

        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 50.0 * position_cost + 10.0 * orientation_cost + 0.0 * velocity_cost
