from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task, ControlMode


class ParticleCollision(Task):
    """A velocity-controlled 3D point mass that reaches a fixed target while avoiding an obstacle."""

    def __init__(
        self,
        planning_horizon: int = 5,
        sim_steps_per_control_step: int = 5,
        control_mode: ControlMode = ControlMode.GENERAL,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/particle_collision/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["particle"],
            control_mode=control_mode,
        )

        # Setup config (reusing relevant parts from Particle3D)
        self.config = {
            "p_min": 5.0,
            "p_max": 30.0,
            "d_min": 1.0,
            "d_max": 10.0,
            "trans_p_min": 5.0,
            "trans_p_max": 30.0,
            "rot_p_min": 5.0,
            "rot_p_max": 30.0,
            "trans_p": 300.0,
            "rot_p": 50.0,
            "pos_min": [-0.3, -0.3, -0.3],
            "pos_max": [0.3, 0.3, 0.3],
            "rot_min": [-3.14, -3.14, -3.14],
            "rot_max": [3.14, 3.14, 3.14],
        }

        self.particle_id = mj_model.site("particle").id
        self.reference_id = mj_model.site("reference").id
        self.target_pos_id = mj_model.site("target").pos
        self.obstacle_radius = 0.1

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) includes obstacle avoidance."""
        # Get current particle position and obstacle position
        particle_pos = state.site_xpos[self.particle_id]
        obstacle_pos = state.mocap_pos[0]
        target_pos = state.mocap_pos[1]

        # Distance to obstacle
        dist_to_obstacle = jnp.linalg.norm(particle_pos - obstacle_pos)

        # Soft constraint for obstacle avoidance (barrier function)
        obstacle_cost = jnp.maximum(0, self.obstacle_radius - dist_to_obstacle) ** 2

        # Distance to target
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - target_pos)
        )

        # Control cost
        control_cost = jnp.sum(jnp.square(state.ctrl - particle_pos))

        # Velocity cost
        velocity_cost = jnp.sum(jnp.square(state.qvel))

        return (
            1e2 * position_cost
            + 1e6 * obstacle_cost
            + 1e1 * control_cost
            + 5e1 * velocity_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T) with higher weight on target reaching."""
        return 2.0 * self.running_cost(state, None)
