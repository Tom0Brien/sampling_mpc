#!/usr/bin/env python3
"""
GPU-Accelerated Cartesian Pose Impedance Controller using MuJoCo MJX.
Simulates K parallel instances with perturbed goal poses.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx
from typing import Tuple, List, Optional
from mujoco.mjx._src.support import jac
from util import *
import os 

# Set XLA flags for better performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "

def impedance_control_mjx(
    model_mjx,
    data_mjx,
    p_des: jnp.ndarray, 
    eul_des: jnp.ndarray,
    Kp: jnp.ndarray,
    Kd: jnp.ndarray,
    nullspace_stiffness: float,
    q_d_nullspace: jnp.ndarray,
    site_id: int = None
) -> jnp.ndarray:
    """
    Compute Cartesian pose impedance control torque using MJX.
    Designed to work with JAX transformations (jit, vmap).
    """
    # Run forward dynamics
    data_mjx = mjx.forward(model_mjx, data_mjx)
    q  = data_mjx.qpos
    dq = data_mjx.qvel
    
    # End-effector pose
    p_curr = data_mjx.site_xpos[site_id]
    rot_ee = data_mjx.site_xmat[site_id].reshape((3, 3))
    quat_curr = mat_to_quat(rot_ee)

    # Get jacobians for the site
    jacp, jacr = jac(model_mjx, data_mjx, p_curr, 8)
    jacp = jacp.T
    jacr = jacr.T
    J = jnp.concatenate([jacp, jacr], axis=0)

    # Compute positional/orientation errors
    e_pos = p_curr - p_des
    e_ori = orientation_error(quat_curr, eul_to_quat(eul_des), rot_ee)
    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # End-effector velocity in task space
    v = J @ dq

    # Cartesian impedance
    F_ee_des = -Kp @ e - Kd @ v
    tau_task = J.T @ F_ee_des

    # Nullspace control
    Jt_pinv = pseudo_inverse(J.T)
    proj = jnp.eye(model_mjx.nv) - (J.T @ Jt_pinv)
    dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Coriolis Compensation (no gravity compensation in Franka example)
    tau_cor = data_mjx.qfrc_bias - data_mjx.qfrc_gravcomp

    return tau_task + tau_cor + tau_null


def run_parallel_simulations(
    model,
    K: int,
    N: int,
    noise_scale: float = 0.05,
    p_des: np.ndarray = np.array([0.7, 0.0, 0.3]),
    eul_des: np.ndarray = np.array([0.0, 3.14, 0.0]),
    dt: float = None,
    seed: int = 42
) -> Tuple[List[List[mujoco.MjData]], np.ndarray, np.ndarray]:
    """
    Run K parallel simulations with perturbed goal poses for N steps.
    
    Args:
        model: MuJoCo model
        K: Number of parallel simulations
        N: Number of steps per simulation
        noise_scale: Scale of Gaussian noise for goal perturbation
        p_des: Base desired position
        eul_des: Base desired orientation (Euler angles)
        dt: Timestep (if None, use model.opt.timestep)
        seed: Random seed
        
    Returns:
        rollouts: List of K lists, each containing N MjData objects
        p_des_array: Array of K perturbed positions
        eul_des_array: Array of K perturbed orientations
    """
    if dt is None:
        dt = model.opt.timestep
    
    # Get site ID before MJX conversion
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    
    # Setup MJX model and initial data
    model_mjx = mjx.put_model(model)
    data = mujoco.MjData(model)
    # Set the initial joint positions
    data.qpos[:7] = jnp.array([0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4])
    mujoco.mj_forward(model, data)
    
    # Get initial joint positions for nullspace control
    q_d_nullspace = jnp.array(data.qpos)
    
    # Generate perturbed goal poses
    key = jax.random.PRNGKey(seed)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Position perturbations
    p_noise = jax.random.normal(subkey1, (K, 3)) * noise_scale
    p_des_array = jnp.array(p_des) + p_noise
    
    # Orientation perturbations (small perturbations to Euler angles)
    eul_noise = jax.random.normal(subkey2, (K, 3)) * noise_scale * 0.5  # smaller noise for orientation
    eul_des_array = jnp.array(eul_des) + eul_noise
    
    # Create K copies of the initial data
    data_mjx = mjx.put_data(model, data)
    data_batch = jax.jax.tree.map(lambda x: jnp.repeat(x[None], K, axis=0), data_mjx)
    
    # Cartesian and nullspace gains
    Kp = jnp.diag(
        jnp.array([300, 300, 300, 50, 50, 50], dtype=float)
    )
    # 2 * square root of Kp
    Kd = 2.0 * jnp.sqrt(Kp)
    nullspace_stiffness = 0.01
    
    # Define control function for a single simulation
    def control_step(data, p_des, eul_des):
        tau_d = impedance_control_mjx(
            model_mjx=model_mjx,
            data_mjx=data,
            p_des=p_des,
            eul_des=eul_des,
            Kp=Kp,
            Kd=Kd,
            nullspace_stiffness=nullspace_stiffness,
            q_d_nullspace=q_d_nullspace,
            site_id=site_id
        )
        # Apply control and step
        data = data.replace(ctrl=data.ctrl.at[:model_mjx.nv].set(tau_d))
        data = mjx.step(model_mjx, data)
        return data
    
    # Vectorize the control step across K simulations
    vmap_control_step = jax.vmap(control_step, in_axes=(0, 0, 0))
    
    # JIT compile for efficiency
    jit_vmap_control_step = jax.jit(vmap_control_step)
    
    # Initialize storage for rollouts
    rollouts = [[None for _ in range(N)] for _ in range(K)]
    
    # Run simulations
    print(f"Running {K} parallel simulations for {N} steps...")
    start_time = time.time()
    
    current_batch = data_batch
    for step in range(N):
        # Perform parallel control and stepping
        current_batch = jit_vmap_control_step(current_batch, p_des_array, eul_des_array)
        
        # Store rollout data (convert back to CPU)
        mjx_datas = mjx.get_data(model, current_batch)
        for k in range(K):
            # Create a copy to avoid reference issues
            rollouts[k][step] = mjx_datas[k]
    
    elapsed = time.time() - start_time
    print(f"Simulations completed in {elapsed:.2f} seconds")
    
    return rollouts, np.array(p_des_array), np.array(eul_des_array)


def visualize_rollouts(
    model,
    rollouts: List[List[mujoco.MjData]],
    p_des_array: np.ndarray,
    eul_des_array: np.ndarray,
    render_fps: float = 60.0,
    fixed_camera_id: Optional[int] = None
):
    """
    Visualize multiple rollouts sequentially.
    
    Args:
        model: MuJoCo model
        rollouts: List of K lists, each containing N MjData objects
        p_des_array: Array of K perturbed positions
        eul_des_array: Array of K perturbed orientations
        render_fps: Rendering framerate
        fixed_camera_id: ID of fixed camera to use (None for default)
    """
    K = len(rollouts)
    N = len(rollouts[0])
    
    # Create a viewer
    data = rollouts[0][0]
    pert = mujoco.MjvPerturb()
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        render_period = 1.0 / render_fps

        # Initialize K frames (one for each rollout)
        frame_ids = []
        for k in range(K):
            frame_id = init_desired_pose_frame(viewer)
            frame_ids.append(frame_id)
          
        # Visualize all the rollouts
        for step in range(N):
            loop_start = time.time()

            # Clear the scene
            viewer.user_scn.ngeom = 0
            
            # Reinitialize K frames after clearing the scene
            frame_ids = []
            for k in range(K):
                frame_id = init_desired_pose_frame(viewer)
                frame_ids.append(frame_id)
            
            # Update the non-transparent data
            for field in dir(rollouts[0][step]):
                        if field.startswith('_'):
                            continue
                        try:
                            setattr(data, field, getattr(rollouts[0][step], field))
                        except (AttributeError, TypeError):
                            pass
            # Add geoms for all rollouts
            for k in range(K):
                data_k = rollouts[k][step]
                mujoco.mjv_addGeoms(model, data_k, vopt, pert, catmask, viewer.user_scn)
                update_desired_pose_frame(viewer, frame_ids[k], p_des_array[k], eul_to_quat(eul_des_array[k]))
            viewer.sync()
            
            if not viewer.is_running():
                return
            
            # Control playback rate
            elapsed = time.time() - loop_start
            if elapsed < render_period:
                time.sleep(render_period - elapsed)
        
        # Pause briefly between trajectories
        time.sleep(1.0)

if __name__ == "__main__":
    xml_path = "models/mujoco_menagerie/franka_emika_panda/mjx_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # Set up simulation parameters
    K = 5          # Number of parallel simulations
    N = 250         # Number of steps per simulation
    noise_scale = 0.1  # Scale of Gaussian noise for goal perturbation
    render_fps = 60.0
    
    # Desired pose in position + Euler angles
    p_des = jnp.array([0.5, 0.0, 0.3])
    eul_des = jnp.array([-3.14, 0.0, 0.0])
    
    # Run parallel simulations
    rollouts, p_des_array, eul_des_array = run_parallel_simulations(
        model, K, N, noise_scale, p_des, eul_des
    )
    
    # Visualize the rollouts
    visualize_rollouts(model, rollouts, p_des_array, eul_des_array, render_fps)
