#!/usr/bin/env python3
import time
import threading
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional
from enum import Enum
from dataclasses import dataclass
from mujoco import mjx
import mujoco


class ControllerStatus(Enum):
    """Enum for controller status"""

    IDLE = 0
    PLANNING = 1
    READY = 2
    ERROR = 3


@dataclass
class ControlResult:
    """Container for controller results"""

    action: np.ndarray
    timestamp: float
    cost: Optional[float] = None
    info: Optional[Dict[str, Any]] = None


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
        self.status = ControllerStatus.IDLE

        # Action data
        self.latest_plan = None
        self.plan_timestamp = 0.0

        # Compile functions
        self._compile_functions()

        # Start planning thread
        self._start_planning_thread()

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
                "explore_fraction": 0.5,
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
        self.jit_optimize = jax.jit(self.controller.optimize, donate_argnums=(1,))

        # Compile interpolation function for smooth action application
        self.jit_interp_func = jax.jit(self.controller.interp_func)

    def _start_planning_thread(self):
        """Start the asynchronous planning thread"""

        def planning_worker():
            """Worker function for the planning thread"""
            while not self.should_stop.is_set():
                # Wait for signal to plan or timeout
                self.planning_event.wait(timeout=self.planning_dt)
                self.planning_event.clear()

                # Check if we should stop
                if self.should_stop.is_set():
                    break

                # Do planning
                self._plan()

        # Create and start planning thread
        self.planning_thread = threading.Thread(target=planning_worker, daemon=True)
        self.planning_thread.start()

        # Signal to start planning immediately
        self.planning_event.set()

    def _plan(self):
        """
        Run planning using the latest state data.
        This is called by the planning thread.
        """
        try:
            self.status = ControllerStatus.PLANNING

            # Get the latest state data under the lock
            with self.state_lock:
                # Create a fresh mjx_data from the latest CPU state
                mjx_data = mjx.put_data(self.task.mj_model, self.mj_data)

            # Run optimization with the latest state
            start_time = time.time()
            with self.plan_lock:
                self.policy_params, rollouts = self.jit_optimize(
                    mjx_data, self.policy_params
                )
                self.latest_plan = rollouts
                self.plan_timestamp = time.time()

            plan_time = time.time() - start_time

            # Update status
            self.status = ControllerStatus.READY

            cost = float(jnp.sum(rollouts.costs[0])) if rollouts is not None else None
            print(f"Planning completed in {plan_time:.4f}s")
            if cost is not None:
                print(f"Cost: {cost:.4f}")

        except Exception as e:
            print(f"Error during planning: {e}")
            self.status = ControllerStatus.ERROR

    def update_state(self, state_data):
        """
        Update the internal state with the latest sensor data.
        This function should be called whenever new sensor data is available.

        Args:
            state_data: Hardware-specific state data

        Returns:
            bool: True if state was successfully updated
        """
        with self.state_lock:
            try:
                # Update the CPU state
                self._update_internal_state(state_data)

                # Run forward kinematics to ensure consistency
                mujoco.mj_forward(self.task.mj_model, self.mj_data)

                # Update timestamp
                self.last_state_update_time = time.time()

                # Signal planning thread that new state is available
                self.planning_event.set()

                return True
            except Exception as e:
                print(f"Error updating state: {e}")
                self.status = ControllerStatus.ERROR
                return False

    def _update_internal_state(self, state_data):
        """
        Convert hardware state data to internal MuJoCo state.
        This method must be implemented by subclasses.

        Args:
            state_data: Hardware-specific state data
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def get_action(self, current_time=None):
        """
        Get the control action for the current time by interpolating the latest plan.

        Args:
            current_time: Current time (if None, uses system time)

        Returns:
            ControlResult: Object containing action and metadata
        """
        with self.plan_lock:
            try:
                # Use provided time or system time
                t = current_time if current_time is not None else time.time()

                # If no plan is available yet, return zero action
                if self.latest_plan is None:
                    return ControlResult(
                        action=np.zeros(self.task.model.nu),
                        timestamp=t,
                        cost=None,
                        info={"status": "no_plan"},
                    )

                # Calculate time since planning
                relative_time = t - self.plan_timestamp

                # Interpolate action from the plan
                tq = jnp.array([relative_time])
                tk = self.policy_params.tk
                knots = self.policy_params.mean[None, ...]
                action = self.jit_interp_func(tq, tk, knots)[0][0]

                # Get cost if available - with better error handling
                try:
                    if (
                        self.latest_plan is not None
                        and hasattr(self.latest_plan, "costs")
                        and len(self.latest_plan.costs) > 0
                    ):
                        cost_val = jnp.sum(self.latest_plan.costs[0])
                        # Convert from JAX array to Python float
                        cost = float(cost_val)
                    else:
                        cost = None
                except Exception as e:
                    print(f"Warning: Error calculating cost: {e}")
                    cost = None

                return ControlResult(
                    action=action,
                    timestamp=t,
                    cost=cost,
                    info={"status": "success", "plan_age": relative_time},
                )

            except Exception as e:
                print(f"Error getting action: {e}")
                self.status = ControllerStatus.ERROR
                return ControlResult(
                    action=np.zeros(self.task.model.nu),
                    timestamp=time.time(),
                    cost=None,
                    info={"status": "error", "error": str(e)},
                )

    def run_control_loop(self, hardware_interface, max_iterations=None, duration=None):
        """
        Run the main control loop that updates state, plans, and acts.

        Args:
            hardware_interface: Interface to send commands to hardware
            max_iterations: Maximum number of control iterations (None for infinite)
            duration: Maximum duration in seconds (None for infinite)
        """
        start_time = time.time()
        iteration = 0
        print("Starting control loop")
        try:
            while True:
                loop_start = time.time()

                # Check termination conditions
                if max_iterations and iteration >= max_iterations:
                    print(f"Reached max iterations ({max_iterations})")
                    break

                if duration and (time.time() - start_time) > duration:
                    print(f"Reached max duration ({duration}s)")
                    break

                # Get latest state from hardware
                state_data = hardware_interface.get_state()
                self.update_state(state_data)

                # Get action for current time
                control_result = self.get_action()

                # Send action to hardware
                self.send_command(control_result.action)

                # Sleep to maintain control frequency
                elapsed = time.time() - loop_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)

                # Print status periodically
                if iteration % 10 == 0:  # Print every 10 iterations
                    rtr = self.control_dt / (
                        time.time() - loop_start
                    )  # Real-time ratio

                    # Safer cost formatting
                    cost_str = "N/A"
                    if control_result.cost is not None:
                        try:
                            cost_str = f"{control_result.cost}"
                        except (ValueError, TypeError):
                            cost_str = f"{control_result.cost}"

                    print(f"Iter {iteration}: RTR: {rtr:.2f}, Cost: {cost_str}")

                iteration += 1

        except KeyboardInterrupt:
            print("Control interrupted by user")
        finally:
            # Clean up
            self.stop()

    def stop(self):
        """Stop all running threads and clean up resources"""
        # Signal threads to stop
        self.should_stop.set()
        self.planning_event.set()  # Wake up planning thread

        # Wait for planning thread to finish
        if self.planning_thread is not None and self.planning_thread.is_alive():
            self.planning_thread.join(timeout=1.0)

        print("Controller stopped")

    def set_goal(self, goal):
        """
        Set a new goal for the controller.
        This method must be implemented by subclasses.

        Args:
            goal: Task-specific goal representation
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def send_command(self, action):
        """
        Send a command to the hardware.
        This method must be implemented by subclasses.

        Args:
            action: Control action to send
        """
        raise NotImplementedError("This method must be implemented by subclasses")
