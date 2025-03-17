from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mppii import ROOT
from mppii.task_base import Task


class ParticleCorridor(Task):
    """A particle pushes a box through a narrow corridor without touching the walls."""

    def __init__(
        self,
        planning_horizon: int = 5,
        sim_steps_per_control_step: int = 5,
        optimize_gains: bool = False,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/particle_corridor/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["particle", "box_site"],
            optimize_gains=optimize_gains,
        )

        self.particle_id = mj_model.site("particle").id
        self.reference_id = mj_model.site("reference").id
        self.corridor_center_id = mj_model.site("corridor_center").id

        # Sensor IDs for touch sensors
        self.box_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "box_position"
        )
        self.left_wall_touch_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wall_touch"
        )
        self.right_wall_touch_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wall_touch"
        )

        # Define parameters for wall collision detection
        self.wall_collision_penalty = 1e3  # Penalty for wall collisions

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

    def _get_sensor_data(self, state: mjx.Data, sensor_id: int) -> jax.Array:
        """Gets sensor data given sensor ID."""
        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]
        return state.sensordata[sensor_adr : sensor_adr + sensor_dim]

    def _wall_collision_cost(self, state: mjx.Data) -> jax.Array:
        """Calculate cost for box collision with walls using touch sensors.

        The touch sensor returns the normal force on the sensor site.
        A non-zero value indicates contact.
        """
        # Get touch data from sensors
        left_wall_touch = self._get_sensor_data(state, self.left_wall_touch_sensor)
        right_wall_touch = self._get_sensor_data(state, self.right_wall_touch_sensor)

        # Calculate magnitude of touch forces
        left_touch_magnitude = jnp.sum(jnp.square(left_wall_touch))
        right_touch_magnitude = jnp.sum(jnp.square(right_wall_touch))

        # Combine touch magnitudes with quadratic penalty
        total_touch = left_touch_magnitude + right_touch_magnitude

        return total_touch

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost encourages target tracking and avoiding wall collisions."""
        # Particle tracking cost
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )

        # Control cost
        control_cost = jnp.sum(
            jnp.square(jnp.array([control[0], control[1], 0]) - state.mocap_pos[0])
        )

        # Wall collision cost based on touch sensors
        wall_cost = self._wall_collision_cost(state)

        return (
            1e1 * position_cost
            + self.wall_collision_penalty * wall_cost
            + 1e0 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )
        wall_cost = self._wall_collision_cost(state)

        return 1e1 * position_cost + 1e6 * wall_cost

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
