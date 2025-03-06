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
from util import init_desired_pose_frame, update_desired_pose_frame
    
# JAX utility functions for impedance control
def quat_mul(q1, q2):
    """Hamilton product of two quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return jnp.array([w, x, y, z])

def quat_flip_if_needed(q_curr, q_des):
    """Flip q_curr sign if needed to avoid discontinuity."""
    return jnp.where(jnp.dot(q_des, q_curr) < 0.0, -q_curr, q_curr)
def orientation_error(q_curr, q_des, R_world_ee):
    """Compute orientation error in base frame."""
    q_curr_flipped = quat_flip_if_needed(q_curr, q_des)
    w, x, y, z = q_curr_flipped
    q_inv = jnp.array([w, -x, -y, -z])
    q_err = quat_mul(q_inv, q_des)
    e_local = q_err[1:]
    return -R_world_ee @ e_local


def quaternion_conjugate(q):
    """Return the conjugate of a quaternion"""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return jnp.array([w, x, y, z])


def euler_to_quat(euler):
    """Convert Euler angles to quaternion"""
    # Assuming euler angles are in XYZ order (roll, pitch, yaw)
    roll, pitch, yaw = euler
    
    cr, sr = jnp.cos(roll/2.0), jnp.sin(roll/2.0)
    cp, sp = jnp.cos(pitch/2.0), jnp.sin(pitch/2.0)
    cy, sy = jnp.cos(yaw/2.0), jnp.sin(yaw/2.0)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return jnp.array([w, x, y, z])


def pseudo_inverse(J, eps=1e-6):
    """Compute the pseudo-inverse of a matrix with regularization"""
    m, n = J.shape
    if m >= n:
        return jnp.linalg.solve(J.T @ J + eps * jnp.eye(n), J.T)
    else:
        return jnp.linalg.solve(J @ J.T + eps * jnp.eye(m), J).T


def impedance_control_mjx(
    model_mjx,
    data_mjx,
    p_des: jnp.ndarray, 
    eul_des: jnp.ndarray,
    cartesian_stiffness: jnp.ndarray,
    cartesian_damping: jnp.ndarray,
    nullspace_stiffness: float,
    q_d_nullspace: jnp.ndarray,
    body_id: int = None
) -> jnp.ndarray:
    """
    Compute Cartesian pose impedance control torque using MJX.
    Designed to work with JAX transformations (jit, vmap).
    """
    # Run forward dynamics
    data_mjx = mjx.forward(model_mjx, data_mjx)
    
    # Joint states
    q = data_mjx.qpos
    dq = data_mjx.qvel

    # End-effector pose
    p_curr = data_mjx.xpos[body_id]
    rot_ee = data_mjx.xmat[body_id].reshape((3, 3))
    quat_curr = data_mjx.xquat[body_id]

    # Get jacobians of body frame (NV, 3)
    jacp, jacr = jac(model_mjx, data_mjx, p_curr, body_id)
    jacp = jacp.T
    jacr = jacr.T
    J = jnp.concatenate([jacp, jacr], axis=0)

    # Compute positional/orientation errors
    e_pos = p_curr - p_des
    e_ori = orientation_error(quat_curr, euler_to_quat(eul_des), rot_ee)
    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # End-effector velocity in task space
    v = jnp.concatenate([jacp @ dq, jacr @ dq], axis=0)

    # Cartesian impedance
    F_ee_des = -cartesian_stiffness @ e - cartesian_damping @ v
    tau_task = J.T @ F_ee_des

    # Nullspace control
    # Jt_pinv = pseudo_inverse(J.T)
    # proj = jnp.eye(7) - (J.T @ Jt_pinv)
    # dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    # tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Coriolis + Gravity Compensation
    cor_grav = data_mjx.qfrc_bias

    return tau_task  + cor_grav #+ tau_null


def run_parallel_simulations(
    model,
    K: int,
    N: int,
    noise_scale: float = 0.05,
    base_p_des: np.ndarray = np.array([0.7, 0.0, 0.3]),
    base_eul_des: np.ndarray = np.array([0.0, 3.14, 0.0]),
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
        base_p_des: Base desired position
        base_eul_des: Base desired orientation (Euler angles)
        dt: Timestep (if None, use model.opt.timestep)
        seed: Random seed
        
    Returns:
        rollouts: List of K lists, each containing N MjData objects
        p_des_array: Array of K perturbed positions
        eul_des_array: Array of K perturbed orientations
    """
    if dt is None:
        dt = model.opt.timestep
    
    # Get body ID before MJX conversion
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # Setup MJX model and initial data
    model_mjx = mjx.put_model(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Get initial joint positions for nullspace control
    q_d_nullspace = jnp.array(data.qpos[:7])
    
    # Generate perturbed goal poses
    key = jax.random.PRNGKey(seed)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Position perturbations
    p_noise = jax.random.normal(subkey1, (K, 3)) * noise_scale
    p_des_array = jnp.array(base_p_des) + p_noise
    
    # Orientation perturbations (small perturbations to Euler angles)
    eul_noise = jax.random.normal(subkey2, (K, 3)) * noise_scale * 0.5  # smaller noise for orientation
    eul_des_array = jnp.array(base_eul_des) + eul_noise
    
    # Create K copies of the initial data
    data_mjx = mjx.put_data(model, data)
    data_batch = jax.tree_map(lambda x: jnp.repeat(x[None], K, axis=0), data_mjx)
    
    # Cartesian and nullspace gains
    cartesian_stiffness = jnp.diag(
        jnp.array([2000, 2000, 2000, 500, 500, 500], dtype=float)
    )
    cartesian_damping = jnp.diag(
        jnp.array([100, 100, 100, 10, 10, 10], dtype=float)
    )
    nullspace_stiffness = 0.0
    
    # Define control function for a single simulation
    def control_step(data, p_des, eul_des):
        tau_d = impedance_control_mjx(
            model_mjx=model_mjx,
            data_mjx=data,
            p_des=p_des,
            eul_des=eul_des,
            cartesian_stiffness=cartesian_stiffness,
            cartesian_damping=cartesian_damping,
            nullspace_stiffness=nullspace_stiffness,
            q_d_nullspace=q_d_nullspace,
            body_id=body_id
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
    with mujoco.viewer.launch_passive(model, rollouts[0][0]) as viewer:
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        render_period = 1.0 / render_fps

        frame_id = init_desired_pose_frame(viewer)
        
        # Visualize each trajectory sequentially
        for k in range(K):
            print(f"Visualizing trajectory {k+1}/{K}")
            
            # Show the desired pose for this trajectory
            print(f"Goal position: {p_des_array[k]}")
            print(f"Goal orientation: {eul_des_array[k]}")
            
            # Visualize this trajectory step by step
            for step in range(N):
                loop_start = time.time()
                
                # Simply render the current state - this matches the passive viewer API
                # The viewer automatically reads from the data object
                viewer.sync()

                update_desired_pose_frame(viewer, frame_id, p_des_array[k], euler_to_quat(eul_des_array[k]))
                
                # Update the data object for the next frame
                if step < N-1:
                    # Copy data from next step to current data
                    for field in dir(rollouts[k][step+1]):
                        if field.startswith('_'):
                            continue
                        try:
                            setattr(rollouts[k][0], field, getattr(rollouts[k][step+1], field))
                        except (AttributeError, TypeError):
                            pass
                
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
    K = 10          # Number of parallel simulations
    N = 200         # Number of steps per simulation
    noise_scale = 0.1  # Scale of Gaussian noise for goal perturbation
    render_fps = 60.0
    
    # Base desired pose
    base_p_des = np.array([0.7, 0.0, 0.3])
    base_eul_des = np.array([0.0, 3.14, 0.0])  # (roll, pitch, yaw) in radians
    
    # Run parallel simulations
    rollouts, p_des_array, eul_des_array = run_parallel_simulations(
        model, K, N, noise_scale, base_p_des, base_eul_des
    )
    
    # Visualize the rollouts
    visualize_rollouts(model, rollouts, p_des_array, eul_des_array, render_fps)
