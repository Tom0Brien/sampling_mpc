#!/usr/bin/env python3
import time
import threading
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util
from typing import Any, Dict, Optional
from enum import Enum
from dataclasses import dataclass
from mujoco import mjx
import mujoco


class HydraxHardwareInterface:
    """
    Generic middleware for connecting Hydrax controllers with real hardware.
    This class handles asynchronous state updates and planning, optimized for
    real-time hardware control.
    """

    def __init__(
        self,
        task,
        controller_type: str = "cem",
        controller_config: Optional[Dict[str, Any]] = None,
        control_frequency: float = 100.0,
        planning_frequency: float = 10.0,
        initial_knots: Optional[jax.Array] = None,
    ):
        """
        Initialize the hardware interface with a Hydrax task and controller.

        Args:
            task: Hydrax task instance
            controller_type: Type of controller ("cem", "mppi", or "ps")
            controller_config: Configuration parameters for the controller
            control_frequency: Frequency at which control actions are sent (Hz)
            planning_frequency: Frequency at which planning is performed (Hz)
        """
        self.task = task
        self.controller_type = controller_type
        self.controller_config = controller_config or {}
        self.control_dt = 1.0 / control_frequency
        self.planning_dt = 1.0 / planning_frequency

        # State on CPU - this is our source of truth for latest state
        self.mj_data = mujoco.MjData(self.task.mj_model)
        mujoco.mj_forward(self.task.mj_model, self.mj_data)  # Initial forward pass

        # Initialize controller
        self._init_controller(initial_knots)

        # State management
        self.state_lock = threading.RLock()
        self.last_state_update_time = 0.0

        # Planning management
        self.plan_lock = threading.RLock()
        self.planning_thread = None
        self.planning_event = threading.Event()
        self.should_stop = threading.Event()

        # Action data
        self.latest_plan = None
        self.plan_timestamp = 0.0

        # Compile functions
        self._compile_functions()

        # Start planning thread
        self._start_planning_thread()

        # Add this to your initialization:
        self.active_policy_params = None

    def _init_controller(self, initial_knots):
        """Initialize the Hydrax controller based on type"""
        from hydrax.algs import MPPI, PredictiveSampling, CEM

        # Get default parameters or use provided ones
        default_params = {
            "num_samples": 512,
            "noise_level": 0.1,
            "plan_horizon": 0.5,
            "spline_type": "zero",
            "num_knots": 11,
        }

        # Override defaults with provided config
        params = {**default_params, **self.controller_config}

        # Create controller based on type
        if self.controller_type.lower() == "mppi":
            self.controller = MPPI(
                self.task,
                num_samples=params["num_samples"],
                noise_level=params["noise_level"],
                plan_horizon=params["plan_horizon"],
                spline_type=params["spline_type"],
                num_knots=params["num_knots"],
            )
        elif self.controller_type.lower() == "ps":
            self.controller = PredictiveSampling(
                self.task,
                num_samples=params["num_samples"],
                noise_level=params["noise_level"],
                plan_horizon=params["plan_horizon"],
                spline_type=params["spline_type"],
                num_knots=params["num_knots"],
            )
        elif self.controller_type.lower() == "cem":
            # Additional CEM-specific parameters
            cem_params = {
                "num_elites": 20,
                "sigma_start": 0.025,
                "sigma_min": 0.005,
                "explore_fraction": 0.0,
            }

            # Override with provided config
            cem_params = {**cem_params, **self.controller_config}

            self.controller = CEM(
                self.task,
                num_samples=params["num_samples"],
                num_elites=cem_params["num_elites"],
                sigma_start=cem_params["sigma_start"],
                sigma_min=cem_params["sigma_min"],
                explore_fraction=cem_params["explore_fraction"],
                plan_horizon=params["plan_horizon"],
                spline_type=params["spline_type"],
                num_knots=params["num_knots"],
            )
        else:
            raise ValueError(f"Unsupported controller type: {self.controller_type}")

        # Initialize policy parameters
        self.policy_params = self.controller.init_params(initial_knots=initial_knots)

    def _compile_functions(self):
        """JIT-compile controller functions for efficiency"""
        # Compile optimize function with donation to avoid memory allocations
        start_time = time.time()
        print("Jitting the controller...")
        self.jit_optimize = jax.jit(self.controller.optimize, donate_argnums=(1,))
        self.jit_get_action = jax.jit(self.controller.get_action)

        # Warm up the JIT functions
        try:
            # First put some data onto GPU
            mjx_data = mjx.put_data(self.task.mj_model, self.mj_data)
            # Run the optimization twice to warm up the JIT stuff
            self.policy_params, _ = self.jit_optimize(mjx_data, self.policy_params)
            self.policy_params, _ = self.jit_optimize(mjx_data, self.policy_params)

            # Set active policy params initially
            self.active_policy_params = self.policy_params

            _ = self.jit_get_action(self.active_policy_params, 0.0)
            _ = self.jit_get_action(self.active_policy_params, 0.0)

            compile_time = time.time() - start_time
            print(f"JIT compilation complete in {compile_time:.4f}s")

        except Exception as e:
            print(f"Warning: JIT warmup failed with error: {e}")

    def _start_planning_thread(self):
        """Start the asynchronous planning thread with improved timing"""

        def planning_worker():
            """Worker function for the planning thread"""
            planning_policy_params = self.policy_params
            self.new_plan_ready = threading.Event()

            mjx_data = mjx.put_data(self.task.mj_model, self.mj_data)

            while not self.should_stop.is_set():
                print("Planning loop")
                try:
                    planning_start = time.time()

                    # Get latest state efficiently
                    with self.state_lock:
                        self.mj_data.time = time.time()

                        st = time.time()
                        print("Putting data onto GPU...")
                        mjx_data = mjx_data.replace(
                            qpos=jnp.array(self.mj_data.qpos),
                            qvel=jnp.array(self.mj_data.qvel),
                            mocap_pos=jnp.array(self.mj_data.mocap_pos),
                            mocap_quat=jnp.array(self.mj_data.mocap_quat),
                            time=self.mj_data.time,
                        )
                        # mjx_data = mjx.put_data(self.task.mj_model, self.mj_data)
                        print(f"Time to put data: {time.time() - st:.4f}s")

                    # Run optimization
                    planning_policy_params, rollouts = self.jit_optimize(
                        mjx_data, planning_policy_params
                    )

                    # Efficient parameter copying using JAX
                    active_params_copy = jax.tree_util.tree_map(
                        lambda x: x.copy(), planning_policy_params
                    )

                    # Atomic update of planning results
                    with self.plan_lock:
                        self.active_policy_params = active_params_copy
                        self.latest_plan = rollouts
                        self.plan_timestamp = self.mj_data.time
                        self.new_plan_ready.set()

                    # Timing management
                    planning_time = time.time() - planning_start
                    next_planning_time = max(
                        self.plan_timestamp + self.planning_dt, time.time()
                    )

                    # Logging
                    if planning_time > self.planning_dt:
                        print(
                            f"Warning: Planning took {planning_time:.4f}s (target: {self.planning_dt:.4f}s)"
                        )
                    print(f"Plan time: {planning_time:.4f}s")
                    # print(f"Cost: {float(jnp.sum(rollouts.costs[0])):.4f}")

                    # Sleep management
                    sleep_time = next_planning_time - time.time()
                    if sleep_time > 0 and not self.should_stop.is_set():
                        time.sleep(sleep_time)

                except Exception as e:
                    print(f"Planning error: {str(e)}")
                    if self.should_stop.is_set():
                        break
                    time.sleep(0.1)  # Brief pause on error

        self.planning_thread = threading.Thread(target=planning_worker, daemon=True)
        self.planning_thread.start()

    def run_control_loop(self, hardware_interface, max_iterations=None):
        """Run the main control loop that updates state, plans, and acts."""
        start_time = time.time()
        iteration = 0
        print("Starting control loop")

        with self.plan_lock:
            current_params = self.active_policy_params

        try:
            while True:
                loop_start = time.time()

                # Check termination conditions
                if max_iterations and iteration >= max_iterations:
                    break

                # Update state and check for new plan
                self.update_state()

                if self.new_plan_ready.is_set():
                    with self.plan_lock:
                        current_params = self.active_policy_params
                        self.new_plan_ready.clear()
                    print(f"Using new plan from time {self.plan_timestamp:.4f}")

                # Execute control action
                if current_params is not None:
                    current_time = jnp.float32(time.time())
                    action = np.array(
                        self.jit_get_action(current_params, current_time),
                        dtype=np.float32,
                    )
                    self.send_command(action)

                # Timing management
                elapsed = time.time() - loop_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
                else:
                    print(f"Control update behind by {elapsed:.4f}s")

                iteration += 1

        except KeyboardInterrupt:
            print("Control interrupted by user")
        finally:
            self.stop()

    def start_debug_viewer(
        self,
        show_traces=False,
        max_traces=5,
        trace_width=5.0,
        trace_color=[1.0, 1.0, 1.0, 0.1],
        fixed_camera_id=None,
    ):
        """
        Start a MuJoCo viewer in a separate thread for debugging.

        Args:
            show_traces: Whether to show traces for planned trajectories
            max_traces: Maximum number of traces to show at once
            trace_width: Width of trace lines
            trace_color: RGBA color of trace lines
            fixed_camera_id: Camera ID to use for fixed view (None for default)
        """
        import mujoco.viewer

        # Define a function to run the viewer in a separate thread
        def viewer_thread_fn():
            with mujoco.viewer.launch_passive(
                self.task.mj_model, self.mj_data
            ) as viewer:
                mujoco.mj_forward(self.task.mj_model, self.mj_data)
                if fixed_camera_id is not None:
                    # Set the custom camera
                    viewer.cam.fixedcamid = fixed_camera_id
                    viewer.cam.type = 2

                # Main viewer loop
                while viewer.is_running() and not self.should_stop.is_set():
                    # Synchronize the viewer with the latest state
                    with self.state_lock:
                        viewer.sync()

                    # Pause to avoid hogging CPU
                    time.sleep(1.0 / 60.0)  # Approx. 60 FPS

        # Create and start the viewer thread
        self.viewer_thread = threading.Thread(target=viewer_thread_fn, daemon=True)
        self.viewer_thread.start()
        print("Debug viewer started. Close the viewer window to stop.")

        return self.viewer_thread

    def update_state(self, state_data):
        """
        Convert generic hardware state data to internal MuJoCo state.
        This method must be implemented by subclasses and update state, including time.

        Args:
            state_data: Hardware-specific state data
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def stop(self):
        """Stop all running threads and clean up resources"""
        # Signal threads to stop
        self.should_stop.set()

        # Wait for planning thread to finish
        if self.planning_thread is not None and self.planning_thread.is_alive():
            self.planning_thread.join(timeout=1.0)

        # Wait for viewer thread if it exists
        if (
            hasattr(self, "viewer_thread")
            and self.viewer_thread is not None
            and self.viewer_thread.is_alive()
        ):
            self.viewer_thread.join(timeout=1.0)

        print("Controller stopped")

    def send_command(self, action):
        """
        Send a command to the hardware.
        """
        raise NotImplementedError("This method must be implemented by subclasses")
