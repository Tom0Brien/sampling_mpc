#!/usr/bin/env python3
"""
Test script to compare CPU and GPU implementations of impedance control.
Both implementations should produce the same torque outputs for identical inputs.
"""

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# Import both controllers
from impedance_control import impedance_control
from impedance_control_gpu import impedance_control_mjx, euler_to_quat

def test_controllers():
    print("Testing CPU vs GPU impedance controllers...")
    
    # Load model
    xml_path = "models/mujoco_menagerie/franka_emika_panda/mjx_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Run forward to initialize
    mujoco.mj_forward(model, data)
    
    # Get body ID
    body_name = "hand"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    # Create MJX versions
    model_mjx = mjx.put_model(model)
    data_mjx = mjx.put_data(model, data)
    
    # Set up control parameters (identical for both)
    p_des = jnp.array([0.7, 0.0, 0.3])
    eul_des = jnp.array([0.0, 3.14, 0.0])
    cartesian_stiffness = jnp.diag(jnp.array([2000, 2000, 2000, 500, 500, 500], dtype=float))
    cartesian_damping = jnp.diag(jnp.array([100, 100, 100, 10, 10, 10], dtype=float))
    nullspace_stiffness = 0.0
    q_d_nullspace = jnp.array(data.qpos[:7])
    
    # Run CPU controller
    tau_cpu = impedance_control(
        model=model,
        data=data,
        body_name=body_name,
        p_des=p_des,
        eul_des=eul_des,
        cartesian_stiffness=cartesian_stiffness,
        cartesian_damping=cartesian_damping,
        nullspace_stiffness=nullspace_stiffness,
        q_d_nullspace=q_d_nullspace
    )
    
    # Run GPU controller
    tau_gpu = impedance_control_mjx(
        model_mjx=model_mjx,
        data_mjx=data_mjx,
        p_des=p_des,
        eul_des=eul_des,
        cartesian_stiffness=cartesian_stiffness,
        cartesian_damping=cartesian_damping,
        nullspace_stiffness=nullspace_stiffness,
        q_d_nullspace=q_d_nullspace,
        body_id=body_id
    )
    
    # Convert to numpy arrays for comparison
    tau_cpu_np = np.array(tau_cpu)
    tau_gpu_np = np.array(tau_gpu)
    
    # Calculate differences
    abs_diff = np.abs(tau_cpu_np - tau_gpu_np)
    rel_diff = abs_diff / (np.abs(tau_cpu_np) + 1e-10)
    
    # Print results
    print("\nCPU controller torques:")
    print(tau_cpu_np)
    print("\nGPU controller torques:")
    print(tau_gpu_np)
    print("\nAbsolute differences:")
    print(abs_diff)
    print("\nRelative differences:")
    print(rel_diff)
    print("\nMax absolute difference:", np.max(abs_diff))
    print("Max relative difference:", np.max(rel_diff))
    
    # Check if differences are within acceptable tolerance
    tolerance = 1e-5
    if np.max(abs_diff) < tolerance:
        print("\n✅ PASS: Controllers produce equivalent outputs within tolerance")
    else:
        print("\n❌ FAIL: Controllers produce different outputs")
        # Find which joints have largest differences
        worst_joint = np.argmax(abs_diff)
        print(f"Largest difference at joint {worst_joint}: {abs_diff[worst_joint]}")

if __name__ == "__main__":
    test_controllers() 