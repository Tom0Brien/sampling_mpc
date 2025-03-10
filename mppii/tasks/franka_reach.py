from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mppii import ROOT
from mppii.task_base import Task
from mppii.util import mat_to_quat, eul_to_quat, orientation_error


class FrankaReach(Task):
    """Franka to reach a target position."""

    def __init__(self, planning_horizon: int = 5, sim_steps_per_control_step: int = 5):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_scene_reach.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["gripper"],
        )

        self.gripper_id = mj_model.site("gripper").id

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        desired_position = jnp.array([0.5, 0.0, 0.5])
        desired_orientation = jnp.array([0.0, -1.0, 0.0, 0.0])  # quat
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.gripper_id] - desired_position)
        )
        # Quaternion difference - compute angular distance between quaternions
        current_rot = state.site_xmat[self.gripper_id].reshape((3, 3))
        current_quat = mat_to_quat(current_rot)
        ori_error = orientation_error(current_quat, desired_orientation, current_rot)
        orientation_cost = jnp.sum(jnp.square(ori_error))

        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 5.0 * position_cost + 2.0 * orientation_cost + 0.1 * velocity_cost
