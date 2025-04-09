#!/usr/bin/env python3
"""
Test script to compare CPU and GPU implementations of impedance control.
Both implementations should produce the same torque outputs for identical inputs.
Tests multiple random configurations to ensure consistency across different states.
"""

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both controllers
from hydrax.controllers.impedance_controllers import (
    impedance_control,
    impedance_control_mjx,
)
from hydrax import ROOT


def test_controllers(num_configs=10):
    print(f"Testing CPU vs GPU impedance controllers with {num_configs} random configurations...")

    # Load model
    xml_path = ROOT + "/models/franka_emika_panda/mjx_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Run forward to initialize
    mujoco.mj_forward(model, data)

    # Get body ID
    body_name = "hand"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

    # Create MJX versions
    model_mjx = mjx.put_model(model)
    
    # Get number of joints and their limits
    nv = model.nv
    qpos_range = []
    for i in range(model.nq):
        qpos_range.append((model.jnt_range[i, 0], model.jnt_range[i, 1]))
    
    # Set up control parameters (identical for both)
    Kp = jnp.diag(jnp.array([2000, 2000, 2000, 500, 500, 500], dtype=float))
    Kd = jnp.diag(jnp.array([100, 100, 100, 10, 10, 10], dtype=float))
    nullspace_stiffness = 0.0
    
    # Storage for max differences
    all_max_abs_diff = []
    all_max_rel_diff = []
    
    # Test with multiple random configurations
    for config_idx in range(num_configs):
        print(f"\n--- Configuration {config_idx+1}/{num_configs} ---")
        
        # Generate random configuration within joint limits
        random_qpos = np.zeros(model.nq)
        for i in range(min(len(qpos_range), model.nq)):
            low, high = qpos_range[i]
            if np.isfinite(low) and np.isfinite(high):
                random_qpos[i] = np.random.uniform(low, high)
            elif np.isfinite(low):
                random_qpos[i] = low + np.random.uniform(0, 1)
            elif np.isfinite(high):
                random_qpos[i] = high - np.random.uniform(0, 1)
            else:
                random_qpos[i] = np.random.uniform(-1, 1)
        
        # Set random velocities (small values for safety)
        random_qvel = np.random.uniform(-0.1, 0.1, model.nv)
        
        # Apply to data
        data.qpos[:] = random_qpos
        data.qvel[:] = random_qvel
        mujoco.mj_forward(model, data)
        
        # Create fresh MJX data with current state
        data_mjx = mjx.put_data(model, data)
        
        # Generate random desired position (within workspace)
        # Using base position plus random offset
        base_pos = data.site_xpos[site_id]
        p_des = jnp.array(base_pos) + jnp.array(np.random.uniform(-0.2, 0.2, 3))
        
        # Random Euler angles (small rotation for safety)
        eul_des = jnp.array(np.random.uniform(-0.5, 0.5, 3))
        
        # Use current configuration for nullspace target
        q_d_nullspace = jnp.array(data.qpos)

        # Run CPU controller
        tau_cpu = impedance_control(
            model=model,
            data=data,
            p_des=p_des,
            eul_des=eul_des,
            Kp=Kp,
            Kd=Kd,
            nullspace_stiffness=nullspace_stiffness,
            q_d_nullspace=q_d_nullspace,
            site_id=site_id,
        )

        # Run GPU controller
        tau_gpu = impedance_control_mjx(
            model_mjx=model_mjx,
            data_mjx=data_mjx,
            p_des=p_des,
            eul_des=eul_des,
            Kp=Kp,
            Kd=Kd,
            nullspace_stiffness=nullspace_stiffness,
            q_d_nullspace=q_d_nullspace,
            site_id=site_id,
        )

        # Convert to numpy arrays for comparison
        tau_cpu_np = np.array(tau_cpu)
        tau_gpu_np = np.array(tau_gpu)

        # Calculate differences
        abs_diff = np.abs(tau_cpu_np - tau_gpu_np)
        rel_diff = abs_diff / (np.abs(tau_cpu_np) + 1e-10)
        
        # Store max differences
        all_max_abs_diff.append(np.max(abs_diff))
        all_max_rel_diff.append(np.max(rel_diff))

        # Print results for this configuration
        print(f"Config {config_idx+1} - Max absolute difference: {np.max(abs_diff):.6f}")
        print(f"Config {config_idx+1} - Max relative difference: {np.max(rel_diff):.6f}")
        
        # Check if differences are within acceptable tolerance
        tolerance = 1e-3
        if np.max(abs_diff) < tolerance:
            print(f"✅ PASS: Config {config_idx+1} within tolerance")
        else:
            print(f"❌ FAIL: Config {config_idx+1} exceeds tolerance")
            # Find which joints have largest differences
            worst_joint = np.argmax(abs_diff)
            print(f"Largest difference at joint {worst_joint}: {abs_diff[worst_joint]}")
            print("CPU torques:", tau_cpu_np)
            print("GPU torques:", tau_gpu_np)
    
    # Overall results
    print("\n--- OVERALL RESULTS ---")
    print(f"Tested {num_configs} random configurations")
    print(f"Highest absolute difference: {np.max(all_max_abs_diff):.6f}")
    print(f"Highest relative difference: {np.max(all_max_rel_diff):.6f}")
    
    # Final pass/fail
    tolerance = 1e-3
    if np.max(all_max_abs_diff) < tolerance:
        print("\n✅ OVERALL PASS: Controllers produce equivalent outputs within tolerance across all configs")
    else:
        print("\n❌ OVERALL FAIL: Controllers produce different outputs in some configurations")
        worst_config = np.argmax(all_max_abs_diff) + 1
        print(f"Worst configuration: {worst_config} with difference {np.max(all_max_abs_diff):.6f}")


if __name__ == "__main__":
    test_controllers(num_configs=10)
