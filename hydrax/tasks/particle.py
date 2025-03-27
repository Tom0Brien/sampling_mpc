from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task, ControlMode


class Particle(Task):
    """A velocity-controlled planar point mass chases a target position."""

    def __init__(
        self,
        planning_horizon: int = 5,
        sim_steps_per_control_step: int = 5,
        control_mode: ControlMode = ControlMode.GENERAL,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/particle/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["particle"],
            control_mode=control_mode,
        )

        # Setup config
        self.config = {
            # Gain limits for GENERAL_VI mode
            "p_min": 5.0,
            "p_max": 30.0,
            "d_min": 1.0,
            "d_max": 10.0,
            # Gain limits for CARTESIAN_SIMPLE_VI mode
            "trans_p_min": 5.0,
            "trans_p_max": 30.0,
            "rot_p_min": 5.0,
            "rot_p_max": 30.0,
            # Fixed gains for CARTESIAN mode
            "trans_p": 300.0,
            "rot_p": 50.0,
            # Control limits for CARTESIAN modes
            "pos_min": [0, -1.0, 0.3],  # x, y, z
            "pos_max": [1.0, 1.0, 1.0],
            "rot_min": [-3.14, -3.14, -3.14],  # roll, pitch, yaw
            "rot_max": [3.14, 3.14, 3.14],
        }

        self.particle_id = mj_model.site("particle").id
        self.reference_id = mj_model.site("reference").id

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        terminal_cost = self.terminal_cost(state)
        return terminal_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )

        # Penalize control effort (distance between reference and particle)
        control_cost = jnp.sum(
            jnp.square(state.ctrl[:2] - state.site_xpos[self.particle_id][:2])
        )

        return 1e2 * position_cost + 1e1 * control_cost

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
