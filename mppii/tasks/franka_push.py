from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mppii import ROOT
from mppii.task_base import Task
from mppii.util import mat_to_quat, eul_to_quat, orientation_error


class FrankaPush(Task):
    """Franka pushing a box to a target pose."""

    def __init__(self, planning_horizon: int = 3, sim_steps_per_control_step: int = 10):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_scene_box_push.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["box_site", "goal"],
        )

        self.gripper_id = mj_model.site("gripper").id
        self.box_id = mj_model.body("box").id
        self.box_site_id = mj_model.site("box_site").id

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages pushing the box to the goal."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # Desired box position
        desired_box_pos = jnp.array([0.7, 0.0, 0.4])
        desired_box_orientation = jnp.array([1.0, 0.0, 0.0, 0.0])
        current_box_pos = state.xpos[self.box_id]
        box_rot = state.xmat[self.box_id].reshape((3, 3))
        box_quat = mat_to_quat(box_rot)
        box_pos_cost = jnp.sum(jnp.square(current_box_pos - desired_box_pos))
        box_orientation_cost = orientation_error(
            box_quat, desired_box_orientation, box_rot
        )

        # Scale gripper-to-box cost based on box's distance to goal
        gripper_pos = state.site_xpos[self.gripper_id]
        box_to_gripper_cost = jnp.sum(jnp.square(gripper_pos - current_box_pos))
        # Higher weight when box is far from goal, lower weight when close
        distance_to_goal = jnp.sqrt(box_pos_cost)
        gripper_cost_scale = 10.0 * jnp.clip(distance_to_goal, 0.0, 1.0)

        # Penalize high velocities
        velocity_cost = jnp.sum(jnp.square(state.qvel))

        return (
            25.0 * box_pos_cost  # Box position
            + 0.0 * box_orientation_cost  # Box orientation
            + 5 * box_to_gripper_cost  # Adaptive gripper-box coupling
            + 0.0 * velocity_cost  # Smooth motion
        )
