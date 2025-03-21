from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task, ControlMode
from hydrax.util import mat_to_quat, eul_to_quat, orientation_error, mat_to_rpy


class FrankaPush(Task):
    """Franka pushing a box to a target pose."""

    def __init__(
        self,
        planning_horizon: int = 10,
        sim_steps_per_control_step: int = 5,
        control_mode: ControlMode = ControlMode.GENERAL,
    ):
        """Load the MuJoCo model and set task parameters.

        Args:
            planning_horizon: The number of control steps (T) to plan over.
            sim_steps_per_control_step: The number of simulation steps per control step.
            control_mode: The control mode to use.
                          CARTESIAN_SIMPLE_VARIABLE_IMPEDANCE is recommended for Franka as it optimizes
                          only translational and rotational p-gains with d-gains automatically set.
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_scene_box_push.xml"
        )

        # Define custom gain limits for this task
        gain_limits = {
            # INDIVIDUAL mode limits
            "p_min": 15.0,
            "p_max": 100.0,
            "d_min": 5.0,
            "d_max": 50.0,
            # SIMPLE mode limits
            "trans_p_min": 15.0,
            "trans_p_max": 100.0,
            "rot_p_min": 15.0,
            "rot_p_max": 100.0,
        }

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["gripper"],
            control_mode=control_mode,
            gain_limits=gain_limits,
        )

        self.gripper_id = mj_model.site("gripper").id
        self.box_id = mj_model.body("box").id
        self.box_site_id = mj_model.site("box_site").id
        self.reference_id = mj_model.site("reference").id

        self.q_d_nullspace = jnp.array(
            [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756, 0, 0]
        )

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
        box_orientation_cost = jnp.sum(
            jnp.square(orientation_error(box_quat, desired_box_orientation, box_rot))
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
            + 10.0 * box_orientation_cost  # Box orientation
            + 40.0 * box_to_gripper_cost  # Close to box cost
            + 0 * gripper_orientation_cost  # Gripper orientation
        )
