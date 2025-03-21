#!/usr/bin/env python3
"""
GPU-Accelerated Cartesian Pose Impedance Controller using MuJoCo MJX.
"""

import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src.support import jac
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
    # Run forward dynamics
    data_mjx = mjx.forward(model_mjx, data_mjx)
    q = data_mjx.qpos
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


def impedance_control(
    model, data, p_des, eul_des, Kp, Kd, nullspace_stiffness, q_d_nullspace, site_id
):
    """
    Compute Cartesian pose impedance control torque using mujoco.
    """
    mujoco.mj_forward(model, data)
    q = jnp.array(data.qpos)
    dq = jnp.array(data.qvel)
    # End-effector pose
    p_curr = jnp.array(data.site_xpos[site_id])
    rot_ee = jnp.array(data.site_xmat[site_id].reshape((3, 3)))
    quat_curr = mat_to_quat(rot_ee)

    # Jacobian
    J = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3, :], J[3:, :], site_id)

    # Compute positional/orientation errors
    e_pos = p_curr - p_des
    e_ori = orientation_error(quat_curr, eul_to_quat(eul_des), rot_ee)
    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # End-effector velocity in task space
    v = J @ dq

    # Cartesian impedance (PD control)
    F_ee_des = -Kp @ e - Kd @ v
    tau_task = J.T @ F_ee_des

    # Nullspace control
    Jt_pinv = pseudo_inverse(J.T)
    proj = jnp.eye(model.nv) - (J.T @ Jt_pinv)
    dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Coriolis Compensation (no gravity compensation in Franka example)
    mujoco.mj_inverse(model, data)
    tau_cor = jnp.array(data.qfrc_bias) - jnp.array(data.qfrc_gravcomp)
    return tau_task + tau_cor + tau_null
