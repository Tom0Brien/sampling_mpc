from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mppii import ROOT
from mppii.task_base import Task


class ParticleBox(Task):
    """A velocity-controlled planar point mass chases a target position while keeping a box in a safe area."""

    def __init__(
        self,
        planning_horizon: int = 5,
        sim_steps_per_control_step: int = 5,
        optimize_gains: bool = False,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/particle_box/scene.xml")

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["particle", "box_site"],
            optimize_gains=optimize_gains,
        )

        self.particle_id = mj_model.site("particle").id
        self.reference_id = mj_model.site("reference").id
        self.box_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "box_position"
        )

        # Define safe area boundaries
        self.safe_area_min = jnp.array([-0.15, -0.15])
        self.safe_area_max = jnp.array([0.15, 0.15])

        # Add safety margin to decrease the safe area
        self.safety_margin = jnp.array([0.02, 0.02])  # 2cm margin on each side
        self.safe_area_min_with_margin = self.safe_area_min + self.safety_margin
        self.safe_area_max_with_margin = self.safe_area_max - self.safety_margin

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
            self.p_gain_min = jnp.ones(mj_model.nu) * 1
            self.p_gain_max = jnp.ones(mj_model.nu) * 50
            self.d_gain_min = jnp.ones(mj_model.nu) * 1
            self.d_gain_max = jnp.ones(mj_model.nu) * 50
            self.u_min = jnp.concatenate([self.u_min, self.p_gain_min, self.d_gain_min])
            self.u_max = jnp.concatenate([self.u_max, self.p_gain_max, self.d_gain_max])

    def _get_box_position(self, state: mjx.Data) -> jax.Array:
        """Get the position of the box."""
        sensor_adr = self.model.sensor_adr[self.box_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 2]  # Only x, y

    def _box_safety_cost(self, state: mjx.Data) -> jax.Array:
        """Calculate cost for box leaving the safe area."""
        box_pos = self._get_box_position(state)

        # Calculate how much the box is outside the safe area (with margin)
        lower_violation = jnp.maximum(0, self.safe_area_min_with_margin - box_pos)
        upper_violation = jnp.maximum(0, box_pos - self.safe_area_max_with_margin)

        # Combine violations
        total_violation = jnp.sum(
            jnp.square(lower_violation) + jnp.square(upper_violation)
        )
        return total_violation

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking and box safety."""
        # Particle tracking cost
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )

        # Control cost: Difference between control position reference and mocap position
        control_cost = jnp.sum(
            jnp.square(jnp.array([control[0], control[1], 0]) - state.mocap_pos[0])
        )

        # Box safety cost
        safety_cost = self._box_safety_cost(state)

        return 1e1 * position_cost + 1e3 * safety_cost + 1e1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )
        safety_cost = self._box_safety_cost(state)

        return 1e1 * position_cost + 1e6 * safety_cost

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
        return {"qpos": data.qpos.at[:2].add(shift)}
