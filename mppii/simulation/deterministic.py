import time
from typing import Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx
import mediapy as media
import matplotlib.pyplot as plt

from mppii.alg_base import SamplingBasedController

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
    show_cost_overlay: bool = True,
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
    """
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.task.planning_horizon} steps "
        f"over a {controller.task.planning_horizon * controller.task.dt} "
        f"second horizon."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params()
    jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # For video recording
    frames = []
    cam = mujoco.MjvCamera()
    # Move the camera 0.5m in the +x direction
    cam.lookat = [0.5, 0.0, 0.5]
    mujoco.mjv_defaultCamera(cam)
    if record_video:
        renderer = mujoco.Renderer(mj_model, height=1080, width=1920)
        # Print in green recording video
        print("\033[92mRecording video...\033[0m")

    # For tracking costs and gains over time
    cost_history = []
    p_gain_history = []
    d_gain_history = []
    control_history = []

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.task.planning_horizon
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

        while viewer.is_running():
            start_time = time.time()

            # Clear previous text overlays
            viewer.user_scn.ngeom = 0

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
            )

            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            # Record cost history
            total_cost = float(jnp.sum(rollouts.costs[0]))
            cost_history.append(total_cost)

            # Visualize the rollouts
            if show_traces:
                ii = 0
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

            # Add cost and gain text overlays
            if show_cost_overlay:
                # Position for cost text - above the scene
                cost_pos = np.array([0.0, 0.0, 0.5])

                # Add cost text
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_LABEL,
                    size=np.array([0.2, 0.2, 0.2]),
                    pos=cost_pos,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1, 1, 1, 1]),  # white text
                )
                geom.label = f"Cost: {total_cost:.4f}"
                viewer.user_scn.ngeom += 1

                # Add gain information if optimizing gains
                if controller.task.optimize_gains:
                    # Extract current gains for display
                    _, current_p_gains, current_d_gains = controller.task.extract_gains(
                        controller.get_action(policy_params, 0)
                    )

                    # Record gain history
                    p_gain_history.append(np.array(current_p_gains))
                    d_gain_history.append(np.array(current_d_gains))

                    # Format gain text
                    p_gain_text = (
                        "P-gains: ["
                        + ", ".join([f"{g:.2f}" for g in current_p_gains])
                        + "]"
                    )
                    d_gain_text = (
                        "D-gains: ["
                        + ", ".join([f"{g:.2f}" for g in current_d_gains])
                        + "]"
                    )

                    # Add P-gain text (positioned below cost text)
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=np.array([0.15, 0.15, 0.15]),
                        pos=cost_pos
                        - np.array([0.0, 0.0, 0.05]),  # position below cost
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.8, 0.8, 1.0, 1]),  # light blue text
                    )
                    geom.label = p_gain_text
                    viewer.user_scn.ngeom += 1

                    # Add D-gain text (positioned below P-gain text)
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=np.array([0.15, 0.15, 0.15]),
                        pos=cost_pos
                        - np.array([0.0, 0.0, 0.1]),  # position below P-gain
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.8, 1.0, 0.8, 1]),  # light green text
                    )
                    geom.label = d_gain_text
                    viewer.user_scn.ngeom += 1

            # Step the simulation
            for i in range(sim_steps_per_replan):
                t = i * mj_model.opt.timestep
                u = controller.get_action(policy_params, t)

                if controller.task.optimize_gains:
                    # Extract control and gains
                    ctrl, p_gains, d_gains = controller.task.extract_gains(u)
                    # Update the actuator gain parameters in the model
                    for j in range(controller.task.nu_ctrl):
                        mj_model.actuator_gainprm[j, 1] = p_gains[j]
                        mj_model.actuator_gainprm[j, 2] = d_gains[j]
                    mj_data.ctrl[: controller.task.nu_ctrl] = np.array(ctrl)
                    control_history.append(np.array(ctrl))
                else:
                    mj_data.ctrl[:] = np.array(u)
                    control_history.append(np.array(u))

                # Extract the first two elements of u as the reference position
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.01, 0, 0],  # Size of the sphere
                    pos=[u[0], u[1], 0.01],  # Position of the sphere
                    mat=np.eye(3).flatten(),
                    rgba=[1.0, 0.0, 0.0, 0.1],  # Red color, explicitly float32
                )
                viewer.user_scn.ngeom += 1

                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                if record_video:
                    renderer.update_scene(mj_data, cam)
                    frames.append(renderer.render())

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s, cost: {total_cost:.4f}",
                end="\r",
            )

    # Preserve the last printout
    print("")

    # Plot cost and gain history if requested
    if plot_costs and cost_history:
        try:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(
                1 + (2 if controller.task.optimize_gains else 0) + 1,
                1,
                figsize=(10, 10),
                sharex=True,
            )

            # If not optimizing gains, axes is not a list, so make it one for consistency
            if not controller.task.optimize_gains:
                axes = [axes] if not isinstance(axes, np.ndarray) else axes

            # Plot cost history
            axes[0].plot(cost_history)
            axes[0].set_title("Total Cost Over Time")
            axes[0].set_ylabel("Cost")
            axes[0].grid(True)

            # Plot gain histories if optimizing gains
            if controller.task.optimize_gains and p_gain_history and d_gain_history:
                # Convert lists of arrays to 2D arrays
                p_gains_array = np.array(p_gain_history)
                d_gains_array = np.array(d_gain_history)

                # Plot P gains
                for i in range(p_gains_array.shape[1]):
                    axes[1].plot(p_gains_array[:, i], label=f"Actuator {i}")
                axes[1].set_title("P Gains Over Time")
                axes[1].set_ylabel("P Gain")
                axes[1].grid(True)
                axes[1].legend()

                # Plot D gains
                for i in range(d_gains_array.shape[1]):
                    axes[2].plot(d_gains_array[:, i], label=f"Actuator {i}")
                axes[2].set_title("D Gains Over Time")
                axes[2].set_ylabel("D Gain")
                axes[2].grid(True)
                axes[2].legend()

                # Plot control signals (last subplot)
                control_plot_idx = 3
            else:
                # If not optimizing gains, control plot is the second subplot
                control_plot_idx = 1

            # Plot control history
            if control_history:
                control_array = np.array(control_history)
                for i in range(control_array.shape[1]):
                    axes[control_plot_idx].plot(
                        control_array[:, i], label=f"Control {i}"
                    )
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

    # Save the video if recording was enabled
    if record_video and frames:
        if video_path is None:
            video_path = f"recordings/simulation_{int(time.time())}.mp4"
        print(f"Saving video to {video_path}...")

        # For realtime playback, FPS should be 1/timestep
        # Since we capture one frame per simulation step
        effective_fps = 1.0 / mj_model.opt.timestep

        print(f"Recording at effective FPS: {effective_fps:.2f} for realtime playback")
        media.write_video(video_path, frames, fps=effective_fps)
        print(f"Video saved successfully!")
