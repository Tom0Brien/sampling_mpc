from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task, ControlMode
from hydrax.util import mat_to_quat, eul_to_quat, orientation_error


class FrankaReach(Task):
    """Franka to reach a target position."""

    def __init__(
        self,
        planning_horizon: int = 5,
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
            ROOT + "/models/franka_emika_panda/mjx_scene_reach.xml"
        )

        # Define custom gain limits for Franka
        gain_limits = {
            # INDIVIDUAL mode limits
            "p_min": 15.0,
            "p_max": 30.0,
            "d_min": 5.0,
            "d_max": 20.0,
            # SIMPLE mode limits
            "trans_p_min": 90.0,
            "trans_p_max": 310.0,
            "rot_p_min": 15.0,
            "rot_p_max": 60.0,
        }

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["gripper"],
            control_mode=control_mode,
            gain_limits=gain_limits,
        )

        self.ee_site_id = mj_model.site("gripper").id
        self.reference_id = mj_model.site("reference").id

        self.q_d_nullspace = jnp.array(
            [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756, 0, 0]
        )

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
            jnp.square(state.site_xpos[self.ee_site_id] - desired_position)
        )
        # Quaternion difference - compute angular distance between quaternions
        current_rot = state.site_xmat[self.ee_site_id].reshape((3, 3))
        current_quat = mat_to_quat(current_rot)
        ori_error = orientation_error(current_quat, desired_orientation, current_rot)
        orientation_cost = jnp.sum(jnp.square(ori_error))

        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 50.0 * position_cost + 10.0 * orientation_cost + 0.0 * velocity_cost
