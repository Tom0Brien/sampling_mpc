#!/usr/bin/env python3
"""
Minimal Cartesian Pose Impedance Controller using MuJoCo + JAX,
with desired orientation specified via Euler angles.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) into a quaternion [w, x, y, z]."""
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return jnp.array([w, x, y, z])

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

def pseudo_inverse(matrix, rtol=1e-4):
    """Compute the Moore-Penrose pseudoinverse."""
    return jnp.linalg.pinv(matrix, rtol=rtol)

def get_body_jacobian(model, data, body_name):
    """Retrieve translational and rotational Jacobian for a body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jnp.array(jacp), jnp.array(jacr)

def quat_to_mat(q):
    """Convert a unit quaternion [w,x,y,z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return jnp.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x**2 + z**2),2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

def init_desired_pose_frame(viewer, num_lines=3):
    """Initialize line geoms to visualize the desired frame."""
    base_id = viewer.user_scn.ngeom
    for _ in range(num_lines):
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 1.0, 1.0, 1.0])
        )
        viewer.user_scn.ngeom += 1
    return base_id

def update_desired_pose_frame(viewer, base_id, x_des, quat_des, length=0.1):
    """Update line geoms to show the desired frame origin and axes."""
    R = quat_to_mat(quat_des)
    axes = {
        0: (jnp.array([1, 0, 0]), np.array([1, 0, 0, 1])),
        1: (jnp.array([0, 1, 0]), np.array([0, 1, 0, 1])),
        2: (jnp.array([0, 0, 1]), np.array([0, 0, 1, 1]))
    }
    for i in range(3):
        axis_unit, color = axes[i]
        g = viewer.user_scn.geoms[base_id + i]
        start_pt = x_des
        end_pt = x_des + R @ axis_unit * length
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=color
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start_pt, end_pt)

def compute_cartesian_pose_impedance_torque(
    model,
    data,
    body_name,
    x_des, quat_des,
    cartesian_stiffness,
    cartesian_damping,
    nullspace_stiffness,
    q_d_nullspace,
    tool_compensation_force,
    activate_tool_compensation,
    tau_prev,
    delta_tau_max
):
    """Compute one step of the Cartesian pose impedance control torque."""
    mujoco.mj_forward(model, data)
    q = jnp.array(data.qpos[:7])
    dq = jnp.array(data.qvel[:7])
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    x_curr = jnp.array(data.xpos[body_id])
    rot_ee = jnp.array(data.xmat[body_id].reshape((3, 3)))
    quat_curr = jnp.array(data.xquat[body_id])

    jacp, jacr = get_body_jacobian(model, data, body_name)
    jacp = jacp[:, :7]
    jacr = jacr[:, :7]
    J = jnp.concatenate([jacp, jacr], axis=0)

    e_pos = x_curr - x_des
    e_ori = orientation_error(quat_curr, quat_des, rot_ee)
    error_6d = jnp.concatenate([e_pos, e_ori], axis=0)

    v_6d = jnp.concatenate([jacp @ dq, jacr @ dq], axis=0)
    F_ee_des = -cartesian_stiffness @ error_6d - cartesian_damping @ v_6d
    tau_task = J.T @ F_ee_des

    Jt_pinv = pseudo_inverse(J.T)
    proj = jnp.eye(7) - (J.T @ Jt_pinv)
    dn = 2.0 * jnp.sqrt(nullspace_stiffness)
    tau_null = proj @ (nullspace_stiffness * (q_d_nullspace - q) - dn * dq)

    tau_tool = jnp.zeros(7)
    if activate_tool_compensation:
        tau_tool = J.T @ tool_compensation_force

    mujoco.mj_inverse(model, data) 
    cor_grav = jnp.array(data.qfrc_bias[:7])
    tau_d_calc = tau_task + tau_null + cor_grav - tau_tool

    diff = tau_d_calc - tau_prev
    tau_d = tau_prev + jnp.clip(diff, -delta_tau_max, delta_tau_max)

    return tau_d

def run_cartesian_pose_impedance_control(
    xml_path="franka_panda.xml",
    sim_time=10.0,
    render_fps=60.0,
    fixed_camera_id=None
):
    """Run a Cartesian pose impedance control demo in MuJoCo."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    steps = int(sim_time / dt)

    # Cartesian and nullspace gains
    cartesian_stiffness = jnp.diag(
        jnp.array([2000, 2000, 2000, 500, 500, 500], dtype=float)
    )
    cartesian_damping   = jnp.diag(
        jnp.array([100, 100, 100, 10, 10, 10], dtype=float)
    )
    nullspace_stiffness = 0.0

    # Tool compensation (inactive by default)
    tool_compensation_force = jnp.zeros(6)
    activate_tool_compensation = False
    delta_tau_max = 100.0

    # Desired pose in position + Euler angles
    x_des = jnp.array([0.7, 0.0, 0.3])

    # Adjust roll, pitch, yaw (in radians) to your preference
    roll, pitch, yaw = 0.0, 3.14, 0.0
    quat_des = euler_to_quat(roll, pitch, yaw)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        frame_id = init_desired_pose_frame(viewer)
        mujoco.mj_forward(model, data)

        # Desired nullspace posture from initial position
        q_d_nullspace = jnp.array(data.qpos[:7])
        tau_prev = jnp.zeros(7)
        render_period = 1.0 / render_fps
        sim_start = time.time()

        for _ in range(steps):
            loop_start = time.time()

            # Compute control torque
            tau_d = compute_cartesian_pose_impedance_torque(
                model=model,
                data=data,
                body_name="hand",
                x_des=x_des,
                quat_des=quat_des,
                cartesian_stiffness=cartesian_stiffness,
                cartesian_damping=cartesian_damping,
                nullspace_stiffness=nullspace_stiffness,
                q_d_nullspace=q_d_nullspace,
                tool_compensation_force=tool_compensation_force,
                activate_tool_compensation=activate_tool_compensation,
                tau_prev=tau_prev,
                delta_tau_max=delta_tau_max
            )

            tau_prev = tau_d
            data.ctrl[:7] = np.array(tau_d)
            mujoco.mj_step(model, data)

            # Update the desired frame visualization
            update_desired_pose_frame(viewer, frame_id, x_des, quat_des)
            viewer.sync()
            if not viewer.is_running():
                break

            elapsed = time.time() - loop_start
            if elapsed < render_period:
                time.sleep(render_period - elapsed)

        print(f"Finished after {time.time() - sim_start:.2f}s of sim-time.")

if __name__ == "__main__":
    xml_file = "models/mujoco_menagerie/franka_emika_panda/mjx_scene.xml"
    run_cartesian_pose_impedance_control(xml_path=xml_file, sim_time=10.0)
