#!/usr/bin/env python3
"""
GPU-Accelerated Cartesian Pose Impedance Controller using MuJoCo MJX.
"""

import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src.support import jac
from mujoco.mjx._src.math import quat_sub, quat_mul, quat_inv
from hydrax.util import *
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
    site_id: int = None,
) -> jnp.ndarray:
    """
    Compute Cartesian pose impedance control torque using MJX.
    Designed to work with JAX transformations (jit, vmap).
    """
    q = data_mjx.qpos
    dq = data_mjx.qvel

    # End-effector pose
    p_curr = data_mjx.site_xpos[site_id]
    quat_curr = mat_to_quat(data_mjx.site_xmat[site_id].reshape((3, 3)))

    # Get jacobians for the site TODO: fix this hardcoded site_id
    jacp, jacr = jac(model_mjx, data_mjx, p_curr, 8)
    J = jnp.concatenate([jacp.T, jacr.T], axis=0)

    # Compute positional/orientation errors
    e_pos = p_des - p_curr
    quat_curr_conj = quat_inv(quat_curr)
    error_quat = quat_mul(eul_to_quat(eul_des), quat_curr_conj)
    e_ori = quat_to_vel(error_quat, 1.0)

    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # End-effector velocity in task space
    v = J @ dq

    # Cartesian impedance
    F_ee_des = Kp @ e - Kd @ v
    tau_task = J.T @ F_ee_des

    # Nullspace control
    Jt_pinv = pseudo_inverse(J.T)
    proj = jnp.eye(model_mjx.nv) - (J.T @ Jt_pinv)
    dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Coriolis compensation only (Gravity compensation added by default in MuJoCo and Franka arms)
    tau_cor = data_mjx.qfrc_bias - data_mjx.qfrc_gravcomp

    return tau_task + tau_cor + tau_null


def impedance_control(
    model, data, p_des, eul_des, Kp, Kd, nullspace_stiffness, q_d_nullspace, site_id
):
    """
    Compute Cartesian pose impedance control torque using mujoco.
    """
    q = jnp.array(data.qpos)
    dq = jnp.array(data.qvel)
    # End-effector pose
    p_curr = jnp.array(data.site_xpos[site_id])
    quat_curr = mat_to_quat(jnp.array(data.site_xmat[site_id].reshape((3, 3))))

    # Jacobian
    J = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3, :], J[3:, :], site_id)

    # Compute positional/orientation errors
    e_pos = p_des - p_curr
    quat_curr_conj = quat_inv(quat_curr)
    error_quat = quat_mul(eul_to_quat(eul_des), quat_curr_conj)
    e_ori = quat_to_vel(error_quat, 1.0)

    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # End-effector velocity in task space
    v = J @ dq

    # Cartesian impedance (PD control)
    F_ee_des = Kp @ e - Kd @ v
    tau_task = J.T @ F_ee_des

    # Nullspace control
    Jt_pinv = pseudo_inverse(J.T)
    proj = jnp.eye(model.nv) - (J.T @ Jt_pinv)
    dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Coriolis compensation only (Gravity compensation added by default in MuJoCo and Franka arms)
    tau_cor = jnp.array(data.qfrc_bias) - jnp.array(data.qfrc_gravcomp)
    return tau_task + tau_cor + tau_null
