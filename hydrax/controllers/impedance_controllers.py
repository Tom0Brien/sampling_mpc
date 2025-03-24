#!/usr/bin/env python3
"""
GPU-Accelerated Cartesian Pose Impedance Controller using MuJoCo MJX.
"""

import numpy as np
import jax.numpy as jnp
from mujoco import mjx
import mujoco
from mujoco.mjx._src.support import jac
from mujoco.mjx._src.math import quat_mul, quat_inv
from hydrax.util import mat_to_quat, quat_to_vel, eul_to_quat, pseudo_inverse
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
    # Unpack states and end-effector pose.
    q = data_mjx.qpos
    dq = data_mjx.qvel
    p_curr = data_mjx.site_xpos[site_id]
    rot_ee = data_mjx.site_xmat[site_id].reshape((3, 3))
    quat_curr = mat_to_quat(rot_ee)

    # Compute jacobian for the site.
    jacp, jacr = jac(model_mjx, data_mjx, p_curr, site_id)
    J = jnp.concatenate([jacp.T, jacr.T], axis=0)

    # Compute positional/orientation errors.
    e_pos = p_des - p_curr
    quat_curr_conj = quat_inv(quat_curr)
    error_quat = quat_mul(eul_to_quat(eul_des), quat_curr_conj)
    e_ori = quat_to_vel(error_quat, 1.0)
    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # Compute end-effector velocity in task space.
    v = J @ dq

    # Compute cartesian impedance (PD control).
    F_ee_des = Kp @ e - Kd @ v
    tau_task = J.T @ F_ee_des

    # Compute nullspace control.
    Jt_pinv = pseudo_inverse(J.T)
    proj = jnp.eye(model_mjx.nv) - (J.T @ Jt_pinv)
    dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Coriolis compensation (Gravity compensation added by default in MuJoCo and Franka arms).
    tau_cor = data_mjx.qfrc_bias - data_mjx.qfrc_gravcomp

    return tau_task + tau_cor + tau_null


def impedance_control(
    model, data, p_des, eul_des, Kp, Kd, nullspace_stiffness, q_d_nullspace, site_id
):
    """
    Compute Cartesian pose impedance control torque using mujoco.
    """
    # Unpack states and end-effector pose.
    q = jnp.array(data.qpos)
    dq = jnp.array(data.qvel)
    p_curr = jnp.array(data.site_xpos[site_id])
    rot_ee = jnp.array(data.site_xmat[site_id].reshape((3, 3)))
    quat_curr = mat_to_quat(rot_ee)

    # Compute jacobian for the site.
    J = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3, :], J[3:, :], site_id)

    # Compute positional/orientation errors.
    e_pos = p_des - p_curr
    quat_curr_conj = quat_inv(quat_curr)
    error_quat = quat_mul(eul_to_quat(eul_des), quat_curr_conj)
    e_ori = quat_to_vel(error_quat, 1.0)
    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # Compute end-effector velocity in task space.
    v = J @ dq

    # Compute cartesian impedance (PD control).
    F_ee_des = Kp @ e - Kd @ v
    tau_task = J.T @ F_ee_des

    # Compute nullspace control.
    Jt_pinv = pseudo_inverse(J.T)
    proj = jnp.eye(model.nv) - (J.T @ Jt_pinv)
    dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Compute coriolis compensation (Gravity compensation added by default in MuJoCo and Franka arms).
    mujoco.mj_inverse(model, data)
    tau_cor = jnp.array(data.qfrc_bias) - jnp.array(data.qfrc_gravcomp)
    return tau_task + tau_cor + tau_null
