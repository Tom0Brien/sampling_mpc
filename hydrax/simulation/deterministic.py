import time
from typing import Sequence, Dict, Callable, Optional

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx
import mediapy as media
import matplotlib.pyplot as plt

from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import ControlMode
from hydrax.controllers.impedance_controllers import impedance_control

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""


def run_interactive(
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    record_video: bool = False,
    video_path: str = None,
    plot_costs: bool = False,
    show_debug_info: bool = True,
    keyboard_control: bool = True,
    keyboard_step_size: float = 0.01,
    mocap_index: int = 0,
    initial_control: jax.Array = None,
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        show_debug_info: Whether to show debug information like costs, gains,
                         and reference positions.
        keyboard_control: Whether to enable keyboard control of the mocap body.
        keyboard_step_size: How far to move the mocap body with each key press.
        mocap_index: Index of the mocap body to control with keyboard (default: 0).
    """
    # Setup simulation parameters
    print(
        f"Planning with {controller.task.planning_horizon} steps "
        f"over a {controller.task.planning_horizon * controller.task.dt} "
        f"second horizon."
    )

    # Calculate control frequency parameters
    sim_steps_per_replan = max(int(1.0 / (frequency * mj_model.opt.timestep)), 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Initialize controller and warmup
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_control=initial_control)
    jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))

    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Setup video recording if enabled
    frames = []
    renderer = None
    cam = mujoco.MjvCamera()
    cam.lookat = [0.5, 0.0, 0.5]
    mujoco.mjv_defaultCamera(cam)

    if record_video:
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
        print("\033[92mRecording video...\033[0m")

    # Initialize history tracking
    cost_history = []
    p_gain_history = []
    d_gain_history = []
    control_history = []

    # Define keyboard callback
    def key_callback(keycode):
        if not keyboard_control:
            return
        key = keycode & 0xFF
        if key == 7:  # LEFT
            mj_data.mocap_pos[mocap_index, 0] -= keyboard_step_size
        elif key == 6:  # RIGHT
            mj_data.mocap_pos[mocap_index, 0] += keyboard_step_size
        elif key == 9:  # UP
            mj_data.mocap_pos[mocap_index, 1] += keyboard_step_size
        elif key == 8:  # DOWN
            mj_data.mocap_pos[mocap_index, 1] -= keyboard_step_size

    # Run the simulation
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as viewer:
        # Setup camera
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Setup visualization traces
        setup_visualization_traces(
            viewer, controller, show_traces, num_traces, trace_width, trace_color
        )

        # Main simulation loop
        while viewer.is_running():
            start_time = time.time()
            viewer.user_scn.ngeom = 0  # Clear previous visualizations

            # Update controller state with current simulation state
            mjx_data = update_controller_state(mjx_data, mj_data)

            # Run planning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start
            total_cost = float(jnp.sum(rollouts.costs[0]))

            # Visualize rollout traces if enabled
            if show_traces:
                visualize_traces(viewer, controller, rollouts, num_traces, trace_width)

            # Run simulation steps until next planning cycle
            for i in range(sim_steps_per_replan):
                t = i * mj_model.opt.timestep
                u = controller.get_action(policy_params, t)
                cost_history.append(total_cost)

                # Apply control based on control mode
                apply_control(
                    controller,
                    mj_model,
                    mj_data,
                    u,
                    control_history,
                    p_gain_history,
                    d_gain_history,
                )

                # Add visualizations if debug info is enabled
                if show_debug_info:
                    add_debug_visualizations(viewer, controller, mj_data, u, total_cost)

                # Step simulation and update viewer
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                # Record video frame if enabled
                if record_video:
                    renderer.update_scene(mj_data, cam)
                    frames.append(renderer.render())

            # Maintain realtime simulation if possible
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print status information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s, cost: {total_cost:.4f}",
                end="\r",
            )

    print("")  # Preserve the last printout

    # Plot history data if requested
    if plot_costs and cost_history:
        plot_simulation_history(
            controller, cost_history, p_gain_history, d_gain_history, control_history
        )

    # Save recorded video if enabled
    if record_video and frames:
        save_video(video_path, frames, mj_model.opt.timestep)


def setup_visualization_traces(
    viewer, controller, show_traces, num_traces, trace_width, trace_color
):
    """Setup visualization traces for the simulation."""
    if not show_traces:
        return

    num_trace_sites = len(controller.task.trace_site_ids)
    for i in range(num_trace_sites * num_traces * controller.task.planning_horizon):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array(trace_color),
        )
        viewer.user_scn.ngeom += 1


def update_controller_state(mjx_data, mj_data):
    """Update the controller state with current simulation state."""
    return mjx_data.replace(
        qpos=jnp.array(mj_data.qpos),
        qvel=jnp.array(mj_data.qvel),
        mocap_pos=jnp.array(mj_data.mocap_pos),
        mocap_quat=jnp.array(mj_data.mocap_quat),
    )


def visualize_traces(viewer, controller, rollouts, num_traces, trace_width):
    """Visualize rollout traces in the scene."""
    ii = 0
    num_trace_sites = len(controller.task.trace_site_ids)
    for k in range(num_trace_sites):
        for i in range(num_traces):
            for j in range(controller.task.planning_horizon):
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[ii],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    trace_width,
                    rollouts.trace_sites[i, j, k],
                    rollouts.trace_sites[i, j + 1, k],
                )
                ii += 1
                viewer.user_scn.ngeom += 1


def apply_control(
    controller, mj_model, mj_data, u, control_history, p_gain_history, d_gain_history
):
    """Apply control based on the control mode."""
    control_mode = controller.task.control_mode

    if control_mode == ControlMode.GENERAL:
        # Apply control directly
        mj_data.ctrl[:] = np.array(u)
        control_history.append(np.array(u))

    elif control_mode == ControlMode.GENERAL_VI:
        # Extract control and gains
        ctrl, p_gains, d_gains = controller.task.extract_gains(u)

        # Update actuator gain parameters
        for j in range(controller.task.nu_ctrl):
            mj_model.actuator_gainprm[j, 0] = p_gains[j]
            mj_model.actuator_biasprm[j, 1] = -p_gains[j]
            mj_model.actuator_biasprm[j, 2] = -d_gains[j]

        # Apply control
        mj_data.ctrl[: controller.task.nu_ctrl] = np.array(ctrl)

        # Store history
        control_history.append(np.array(ctrl))
        p_gain_history.append(np.array(p_gains))
        d_gain_history.append(np.array(d_gains))

    elif control_mode == ControlMode.CARTESIAN:
        # Compute impedance control
        Kp = jnp.diag(jnp.array([300, 300, 300, 50, 50, 50], dtype=float))
        Kd = 2.0 * jnp.sqrt(Kp)
        tau = impedance_control(
            mj_model,
            mj_data,
            u[:3],
            u[3:],
            Kp,
            Kd,
            1.0,
            controller.task.q_d_nullspace,
            controller.task.ee_site_id,
        )

        # Apply control
        mj_data.ctrl[:] = np.array(tau[: mj_model.nu])
        control_history.append(np.array(u))

    elif control_mode == ControlMode.CARTESIAN_SIMPLE_VI:
        # Extract control and gains
        ctrl, p_gains, d_gains = controller.task.extract_gains(u)

        # Apply control
        tau = impedance_control(
            mj_model,
            mj_data,
            ctrl[:3],
            ctrl[3:],
            p_gains * jnp.identity(6),
            d_gains * jnp.identity(6),
            0.0,
            controller.task.q_d_nullspace,
            controller.task.ee_site_id,
        )

        mj_data.ctrl[:] = np.array(tau)
        control_history.append(np.array(ctrl))


def add_debug_visualizations(viewer, controller, mj_data, u, total_cost):
    """Add debug visualizations to the scene."""
    # Visualize reference position if available
    if hasattr(controller.task, "reference_id"):
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        Rbr = mj_data.site_xmat[controller.task.reference_id].reshape(3, 3)
        rRBb = mj_data.site_xpos[controller.task.reference_id]

        if controller.task.nu_ctrl == 2:
            reference_pos = rRBb + Rbr @ np.array([u[0], u[1], 0.0])
        else:
            reference_pos = rRBb + Rbr @ np.array([u[0], u[1], u[2]])

        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            pos=reference_pos,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.0, 0.0, 0.1],
        )
        viewer.user_scn.ngeom += 1

    # Add cost text
    cost_pos = np.array([0.0, 0.0, 0.5])  # Position above the scene
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0.2, 0.2, 0.2]),
        pos=cost_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([1, 1, 1, 1]),
    )
    geom.label = f"Cost: {total_cost:.4f}"
    viewer.user_scn.ngeom += 1


def plot_simulation_history(
    controller, cost_history, p_gain_history, d_gain_history, control_history
):
    """Plot simulation history data."""
    try:
        # Determine number of subplots based on available data
        optimizing_gains = (
            controller.task.control_mode != ControlMode.GENERAL
            and controller.task.control_mode != ControlMode.CARTESIAN
        )
        num_subplots = (
            1 + (2 if optimizing_gains and p_gain_history and d_gain_history else 0) + 1
        )

        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 10), sharex=True)
        axes = [axes] if not isinstance(axes, np.ndarray) else axes

        # Plot cost history
        axes[0].plot(cost_history)
        axes[0].set_title("Total Cost Over Time")
        axes[0].set_ylabel("Cost")
        axes[0].grid(True)

        # Plot gain histories if available
        if optimizing_gains and p_gain_history and d_gain_history:
            p_gains_array = np.array(p_gain_history)
            d_gains_array = np.array(d_gain_history)

            # P gains
            for i in range(p_gains_array.shape[1]):
                axes[1].plot(p_gains_array[:, i], label=f"Actuator {i}")
            axes[1].set_title("P Gains Over Time")
            axes[1].set_ylabel("P Gain")
            axes[1].grid(True)
            axes[1].legend()

            # D gains
            for i in range(d_gains_array.shape[1]):
                axes[2].plot(d_gains_array[:, i], label=f"Actuator {i}")
            axes[2].set_title("D Gains Over Time")
            axes[2].set_ylabel("D Gain")
            axes[2].grid(True)
            axes[2].legend()

            control_plot_idx = 3
        else:
            control_plot_idx = 1

        # Plot control history
        if control_history:
            control_array = np.array(control_history)
            for i in range(control_array.shape[1]):
                axes[control_plot_idx].plot(control_array[:, i], label=f"Control {i}")
            axes[control_plot_idx].set_title("Control Signals Over Time")
            axes[control_plot_idx].set_ylabel("Control Value")
            axes[control_plot_idx].set_xlabel("Control Steps")
            axes[control_plot_idx].grid(True)
            axes[control_plot_idx].legend()

        plt.tight_layout()
        plt.savefig("recordings/cost_gain_history.png")
        plt.show()
        print(
            "Cost, gain, and control history plotted and saved to 'cost_gain_history.png'"
        )
    except ImportError:
        print(
            "Matplotlib not available for plotting. Install with 'pip install matplotlib'"
        )


def save_video(video_path, frames, timestep):
    """Save recorded video frames to file."""
    if not frames:
        return

    if video_path is None:
        video_path = f"recordings/simulation_{int(time.time())}.mp4"
    print(f"Saving video to {video_path}...")

    effective_fps = 1.0 / timestep
    print(f"Recording at effective FPS: {effective_fps:.2f} for realtime playback")
    media.write_video(video_path, frames, fps=effective_fps)
    print(f"Video saved successfully!")
