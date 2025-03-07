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
    body_name,
    p_des, eul_des,
    Kp,
    Kd,
    nullspace_stiffness,
    q_d_nullspace
):
    """
    Compute one step of the Cartesian pose impedance control torque.
    """
    mujoco.mj_forward(model, data)

    # Joint states
    q = jnp.array(data.qpos)
    dq = jnp.array(data.qvel)

    # End-effector pose
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    p_curr = jnp.array(data.xpos[body_id])
    rot_ee = jnp.array(data.xmat[body_id].reshape((3, 3)))
    quat_curr = jnp.array(data.xquat[body_id])

    # Body Jacobians
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    J = jnp.concatenate([jacp, jacr], axis=0)

    # Compute positional/orientation errors
    e_pos = p_curr - p_des
    e_ori = orientation_error(quat_curr, euler_to_quat(eul_des), rot_ee)
    e = jnp.concatenate([e_pos, e_ori], axis=0)

    # End-effector velocity in task space
    v = jnp.concatenate([jacp @ dq, jacr @ dq], axis=0)

    # Cartesian impedance
    F_ee_des = -Kp @ e - Kd @ v
    tau_task = J.T @ F_ee_des

    # Nullspace control
    # Jt_pinv = pseudo_inverse(J.T)
    # proj = jnp.eye(model.nv) - (J.T @ Jt_pinv)
    # dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    # tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    # Coriolis + Gravity Compensation
    mujoco.mj_inverse(model, data)
    cor_grav = jnp.array(data.qfrc_bias[:model.nv])

    return tau_task + cor_grav  #+ tau_null

if __name__ == "__main__":
    xml_path= "models/mujoco_menagerie/franka_emika_panda/mjx_scene.xml"
    sim_time=10.0
    render_fps=60.0
    fixed_camera_id=None
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    steps = int(sim_time / float(dt)) 

    # Cartesian and nullspace gains
    Kp = jnp.diag(
        jnp.array([2000, 2000, 2000, 500, 500, 500], dtype=float)
    )
    Kd = jnp.diag(
        jnp.array([100, 100, 100, 10, 10, 10], dtype=float)
    )
    nullspace_stiffness = 0.0

    # Desired pose in position + Euler angles
    p_des = jnp.array([0.7, 0.0, 0.3])
    eul_des = jnp.array([0.0, 3.14, 0.0])  # (roll, pitch, yaw) in radians

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        frame_id = init_desired_pose_frame(viewer)
        mujoco.mj_forward(model, data)

        # Desired nullspace posture from initial position
        q_d_nullspace = jnp.array(data.qpos[:7])

        render_period = 1.0 / render_fps
        sim_start = time.time()

        for _ in range(steps):
            loop_start = time.time()

            # Compute control torque
            tau_d = impedance_control(
                model=model,
                data=data,
                body_name="hand",
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
            update_desired_pose_frame(viewer, frame_id, p_des, euler_to_quat(eul_des))
            viewer.sync()
            if not viewer.is_running():
                break

            elapsed = time.time() - loop_start
            if elapsed < render_period:
                time.sleep(render_period - elapsed)

        print(f"Finished after {time.time() - sim_start:.2f}s of sim-time.")