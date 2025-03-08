#!/usr/bin/env python3
"""
Minimal Cartesian Pose Impedance Controller using MuJoCo + JAX,
with desired orientation specified via Euler angles.

Simplified to remove torque rate limiting (delta-tau) and tool compensation.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

from util import *

def impedance_control(
    model,
    data,
    site_id,
    p_des, 
    eul_des,
    Kp,
    Kd,
    nullspace_stiffness,
    q_d_nullspace
):
    """
    Compute one step of the Cartesian pose impedance control torque.
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

if __name__ == "__main__":
    xml_path= "models/franka_emika_panda/mjx_scene.xml"
    sim_time=20.0
    render_fps=120.0
    fixed_camera_id=None
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    steps = int(sim_time / float(dt)) 

    # Cartesian and nullspace gains
    Kp = jnp.diag(
        jnp.array([300, 300, 300, 50, 50, 50], dtype=float)
    )
    # 2 * square root of Kp
    Kd = 2.0 * jnp.sqrt(Kp)
    nullspace_stiffness = 0.01

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    print("body_id: ", body_id)
    # Get the site ID once
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    
    # Desired pose in position + Euler angles
    p_des = jnp.array([0.5, 0.0, 0.3])
    eul_des = jnp.array([-3.14, 0.0, 0.0])

    # Set the initial joint positions
    data.qpos[:7] = jnp.array([0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        frame_id = init_desired_pose_frame(viewer)
        mujoco.mj_forward(model, data)

        # Desired nullspace posture from initial position
        q_d_nullspace = jnp.array(data.qpos)

        render_period = 1.0 / render_fps
        sim_start = time.time()

        for _ in range(steps):
            loop_start = time.time()

            # Compute control torque
            tau_d = impedance_control(
                model=model,
                data=data,
                site_id=gripper_site_id,
                p_des=p_des,
                eul_des=eul_des,
                Kp=Kp,
                Kd=Kd,
                nullspace_stiffness=nullspace_stiffness,
                q_d_nullspace=q_d_nullspace
            )

            # Apply control
            data.ctrl = np.array(tau_d)
            mujoco.mj_step(model, data)

            # Update the desired frame visualization
            update_desired_pose_frame(viewer, frame_id, p_des, eul_to_quat(eul_des))
            viewer.sync()
            if not viewer.is_running():
                break

            elapsed = time.time() - loop_start
            if elapsed < render_period:
                time.sleep(render_period - elapsed)

        print(f"Finished after {time.time() - sim_start:.2f}s of sim-time.")