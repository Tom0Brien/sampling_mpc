#!/usr/bin/env python3
"""
Task-Space Impedance Control (Franka Panda) + Interactive MuJoCo Viewer
With an added coordinate frame to visualize the desired pose.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer  # For the viewer context

###############################################################################
# Quaternion Utilities
###############################################################################

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return jnp.array([w, x, y, z])

def quat_conjugate(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def orientation_error(q_curr, q_des):
    """
    Compute a small orientation error vector from current to desired quaternion.
    For small errors, we often use: 2 * (q_err[xyz]) where q_err = q_des * conj(q_curr).
    """
    q_err = quat_mul(q_des, quat_conjugate(q_curr))
    return 2.0 * q_err[1:]  # the xyz part

def quat_to_mat(q):
    """
    Convert a unit quaternion q = [w, x, y, z] to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    # This is a standard formula for a rotation matrix from a quaternion
    R = jnp.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])
    return R

###############################################################################
# Jacobian Utilities
###############################################################################

def get_body_jacobian(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jnp.array(jacp), jnp.array(jacr)

###############################################################################
# Helper: Initialize and update line geoms for a coordinate frame
###############################################################################

def init_desired_pose_frame(viewer, num_lines=3):
    """
    Add 'num_lines' line geoms to the viewer's user scene.
    Returns the base index in user_scn.geoms so we can update them each frame.
    """
    base_id = viewer.user_scn.ngeom
    for _ in range(num_lines):
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        viewer.user_scn.ngeom += 1
    return base_id

def update_desired_pose_frame(viewer, base_id, x_des, quat_des, length=0.1):
    """
    Update the 3 line geoms to visualize a coordinate frame at (x_des, quat_des).
    Red = X axis, Green = Y, Blue = Z, each of length 'length'.
    """
    # Convert desired quaternion to rotation matrix
    R = quat_to_mat(quat_des)

    # Axis definitions
    axes = {
        0: (jnp.array([1,0,0]), np.array([1,0,0,1])),  # X -> red
        1: (jnp.array([0,1,0]), np.array([0,1,0,1])),  # Y -> green
        2: (jnp.array([0,0,1]), np.array([0,0,1,1]))   # Z -> blue
    }

    for i in range(3):
        axis_unit, color = axes[i]
        g = viewer.user_scn.geoms[base_id + i]

        # Starting point = x_des
        start_pt = x_des
        # End point = x_des + R * axis_unit * length
        end_pt = x_des + R @ axis_unit * length

        # Re-init the line geom with new endpoints and color
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=color
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            2.0,                  # line width in pixels
            start_pt,             # line start
            end_pt                # line end
        )

###############################################################################
# Main Function: Impedance Control + Viewer
###############################################################################

def run_interactive_impedance_control(
    xml_path: str = "franka_panda.xml",
    sim_time: float = 10.0,
    render_fps: float = 60.0,
    fixed_camera_id: int = None,
):
    """
    Runs a MuJoCo simulation of a Franka arm with a simple task-space
    impedance controller, while displaying an interactive viewer.
    Also draws a small coordinate frame at the 'desired pose'.
    """
    # Load model & data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    steps = int(sim_time / dt)

    # Desired EE pose
    x_des = jnp.array([0.5, 0.0, 0.6])         # [m]
    quat_des = jnp.array([0.0, 1.0, 0.0, 0.0])  # wxyz

    # Impedance Gains
    Kp = 500.0
    Kd = 50.0

    # We'll run at a chosen "render_fps" for display
    render_period = 1.0 / render_fps

    # Start the viewer in a context manager
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Optionally fix a camera
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Initialize 3 lines to represent the desired frame
        frame_base_id = init_desired_pose_frame(viewer, num_lines=3)

        sim_start_walltime = time.time()
        for step_i in range(steps):
            loop_start = time.time()

            #--------------------
            # 1) Forward the model (for up-to-date kinematics)
            #--------------------
            mujoco.mj_forward(model, data)

            #--------------------
            # 2) Current end-effector pose
            #--------------------
            body_name = "hand"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            x_curr = jnp.array(data.xpos[body_id])
            print("x_curr: ", x_curr)
            quat_curr = jnp.array(data.xquat[body_id])

            #--------------------
            # 3) Task-space error
            #--------------------
            e_pos = x_des - x_curr
            e_ori = orientation_error(quat_curr, quat_des)
            e_task = jnp.concatenate([e_pos, e_ori], axis=0)
            print("e_task: ", e_task)

            #--------------------
            # 4) Jacobians
            #--------------------
            jacp, jacr = get_body_jacobian(model, data, body_name)
            jacp = jacp[:, :7]  # if your arm has 7 dofs
            jacr = jacr[:, :7]
            J = jnp.concatenate([jacp, jacr], axis=0)

            #--------------------
            # 5) Velocity in task space
            #--------------------
            qvel = jnp.array(data.qvel[:7])
            v_pos = jacp @ qvel
            v_ori = jacr @ qvel
            v_task = jnp.concatenate([v_pos, v_ori], axis=0)

            #--------------------
            # 6) Impedance PD
            #--------------------
            F_task = Kp * e_task - Kd * v_task
            tau = J.T @ F_task
            print("tau: ", tau)

            #--------------------
            # 7) Cancel bias terms
            #--------------------
            mujoco.mj_inverse(model, data)
            tau_g = jnp.array(data.qfrc_bias[:7])
            tau += tau_g

            #--------------------
            # 8) Set controls & step
            #--------------------
            data.ctrl[:7] = np.array(tau)
            mujoco.mj_step(model, data)

            #--------------------
            # 9) Update the viewer
            #--------------------
            # Update the coordinate frame at the desired pose
            update_desired_pose_frame(viewer, frame_base_id, x_des, quat_des, length=0.1)

            viewer.sync()
            if not viewer.is_running():
                break

            # Attempt a near real-time rate
            elapsed = time.time() - loop_start
            if elapsed < render_period:
                time.sleep(render_period - elapsed)

        sim_end_walltime = time.time()
        print(f"Simulation ended after {sim_end_walltime - sim_start_walltime:.2f} seconds.")

###############################################################################
# Entry
###############################################################################

if __name__ == "__main__":
    # Adjust the path to match your local environment
    xml_file = "models/mujoco_menagerie/franka_emika_panda/scene.xml"
    run_interactive_impedance_control(
        xml_path=xml_file,
        sim_time=10.0,
        render_fps=60.0,
        fixed_camera_id=None
    )
