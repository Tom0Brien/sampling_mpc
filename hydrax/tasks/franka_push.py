from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.util import mat_to_quat, eul_to_quat, orientation_error, mat_to_rpy


class FrankaPush(Task):
    """Franka pushing a box to a target pose."""

    def __init__(
        self,
        planning_horizon: int = 10,
        sim_steps_per_control_step: int = 5,
        optimize_gains: bool = False,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_scene_box_push.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["gripper"],
            optimize_gains=optimize_gains,
        )

        self.gripper_id = mj_model.site("gripper").id
        self.box_id = mj_model.body("box").id
        self.box_site_id = mj_model.site("box_site").id
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
            self.p_gain_max = jnp.ones(mj_model.nu) * 100.0
            self.d_gain_min = jnp.ones(mj_model.nu) * 5
            self.d_gain_max = jnp.ones(mj_model.nu) * 50
            self.u_min = jnp.concatenate([self.u_min, self.p_gain_min, self.d_gain_min])
            self.u_max = jnp.concatenate([self.u_max, self.p_gain_max, self.d_gain_max])

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages pushing the box to the goal."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(state.actuator_force))
        return state_cost + 0.001 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # Use mocap position as the desired box position
        desired_box_pos = state.mocap_pos[0]
        # Extract the mocap orientation as desired box orientation
        desired_box_orientation = state.mocap_quat[0]

        current_box_pos = state.xpos[self.box_id]
        box_rot = state.xmat[self.box_id].reshape((3, 3))
        box_quat = mat_to_quat(box_rot)
        box_pos_cost = jnp.sum(jnp.square(current_box_pos - desired_box_pos))
        box_orientation_cost = orientation_error(
            box_quat, desired_box_orientation, box_rot
        )

        # Desired gripper position: 5cm back from box along box-to-goal line
        # Calculate direction vector from box to goal
        box_to_goal = desired_box_pos - current_box_pos
        # Normalize the direction vector
        distance = jnp.linalg.norm(box_to_goal)
        direction = box_to_goal / jnp.maximum(distance, 1e-6)  # Avoid division by zero
        # Calculate desired gripper position: 5cm back from box along this direction
        desired_gripper_pos = current_box_pos - 0.05 * direction  # 5cm offset

        gripper_pos = state.site_xpos[self.gripper_id]
        box_to_gripper_cost = jnp.sum(jnp.square(gripper_pos - desired_gripper_pos))

        # Desired gripper orientation (roll and pitch, yaw should be able to vary)
        gripper_rot = state.site_xmat[self.gripper_id].reshape((3, 3))
        gripper_rpy = mat_to_rpy(gripper_rot)
        gripper_orientation_cost = jnp.sum(
            jnp.square(gripper_rpy[:2] - jnp.array([-3.14, 0.0]))
        )

        return (
            100.0 * box_pos_cost  # Box position
            + 0.0 * box_orientation_cost  # Box orientation
            + 40.0 * box_to_gripper_cost  # Close to box cost
            + 0 * gripper_orientation_cost  # Gripper orientation
        )
