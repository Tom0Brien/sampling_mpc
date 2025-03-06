#!/usr/bin/env python3
"""
A Python version of the Cartesian Pose Impedance Controller logic
demonstrated in the Franka C++ example. Uses MuJoCo for simulation + JAX for math.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

###############################################################################
# Utilities for orientation, pose, Jacobians
###############################################################################

def quat_flip_if_needed(q_curr, q_des):
    """
    Match the logic: 
      if (q_des dot q_curr) < 0, then flip sign of q_curr
    so the quaternions represent the same orientation but avoid
    a discontinuity in the sign.
    """
    if jnp.dot(q_des, q_curr) < 0.0:
        return -q_curr
    return q_curr

def orientation_error(q_curr, q_des, R_world_ee):
    """
    Match the style:
      1) If q_des dot q_curr < 0, flip q_curr sign
      2) error_quat = (orientation.inverse() * orientation_d)
         i.e. q_err = q_curr.conjugate() * q_des, 
         but in the Franka example, they do q_des * q_curr.inverse();
         we invert the logic to replicate exactly the final sign usage.
      3) error_xyz = imaginary part of error_quat
      4) e_ori = -R * error_xyz   (to express error in base frame)
    
    But the Franka code had: 
      error_quaternion = orientation.inverse() * orientation_d_
      error(3..5) = error_quaternion.x, y, z
      error(3..5) = -transform.linear() * error(3..5)

    Where orientation = current, orientation_d_ = desired
    => q_err = q_curr.inverse() * q_des
    => e_ori = - R_curr * q_err[xyz].
    We'll replicate that exactly.
    """
    # Step 1) Possibly flip sign
    q_curr_flipped = quat_flip_if_needed(q_curr, q_des)

    # Step 2) q_err = q_curr.inverse() * q_des
    # "inverse" = conj for unit quaternion
    w, x, y, z = q_curr_flipped
    q_curr_inv = jnp.array([w, -x, -y, -z])  # conj
    # We'll do q_err = q_curr_inv * q_des
    # but in the C++ code they do: orientation.inverse() * orientation_d_
    # which is the same as conj(q_curr)*q_des for unit quats
    q_err = quat_mul(q_curr_inv, q_des)

    # Step 3) The imaginary part of q_err is [x,y,z]
    # That is the local orientation error
    e_local = q_err[1:]  # x, y, z

    # Step 4) Multiply by -R_world_ee
    # in C++: error.tail(3) = -transform.linear() * error.tail(3)
    # transform.linear() is R_world_ee
    e_ori = -R_world_ee @ e_local
    return e_ori

def quat_mul(q1, q2):
    """
    Hamilton product of two quaternions (w,x,y,z).
    Result in wxyz format.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return jnp.array([w, x, y, z])

def pseudo_inverse(matrix, rcond=1e-4):
    """
    Moore-Penrose pseudoinverse, akin to pseudoInverse in the C++ code.
    """
    # jnp.linalg.pinv is fine:
    return jnp.linalg.pinv(matrix, rcond=rcond)

def get_body_jacobian(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jnp.array(jacp), jnp.array(jacr)

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
# Visualization: draw the "desired pose" frame
###############################################################################

def init_desired_pose_frame(viewer, num_lines=3):
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
    # Convert quaternion to rotation matrix
    R = quat_to_mat(quat_des)
    axes = {
        0: (jnp.array([1,0,0]), np.array([1,0,0,1])),  # X -> red
        1: (jnp.array([0,1,0]), np.array([0,1,0,1])),  # Y -> green
        2: (jnp.array([0,0,1]), np.array([0,0,1,1]))   # Z -> blue
    }
    for i in range(3):
        axis_unit, color = axes[i]
        g = viewer.user_scn.geoms[base_id + i]
        start_pt = x_des
        end_pt = x_des + R @ axis_unit * length
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
            2.0,
            start_pt,
            end_pt
        )

###############################################################################
# Main: Cartesian Pose Impedance Controller in Python
###############################################################################

def run_cartesian_pose_impedance_control(
    xml_path="franka_panda.xml",
    sim_time=10.0,
    render_fps=60.0,
    fixed_camera_id=None
):
    """
    Replicates the main logic of the C++ CartesianPoseImpedanceController:
      - Pose error in position + orientation
      - PD in task-space
      - Nullspace PD
      - Coriolis + optional tool compensation
      - Possibly saturate torque rate
    """
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    steps = int(sim_time / dt)

    # 1) Gains for the task space
    #    (like cartesian_stiffness_, cartesian_damping_ in C++)
    #    We'll define them as 6x6 for position/orientation
    #    Use placeholders or configure them as you like
    cartesian_stiffness = jnp.diag(jnp.array([2000, 2000, 2000, 500, 500, 500], dtype=float))
    cartesian_damping   = jnp.diag(jnp.array([100,   100,   100,   10,  10,  10],  dtype=float))

    # 2) Nullspace gains
    nullspace_stiffness = 0.0  # e.g. handle a nominal posture
    # We'll do a small damping ratio of 1
    # => damping_ns = 2 * sqrt(nullspace_stiffness)
    # But let's do it inline

    # 3) Desired pose + orientation (the "position_d_target_" in C++)
    x_des = jnp.array([0.7, 0.0, 0.3])
    quat_des = jnp.array([0.0, 1.0, 0.0, 0.0])  # w, x, y, z

    # 4) Tool compensation (like "activate_tool_compensation_" + "tool_compensation_force_")
    #    We'll assume [fx, fy, fz, 0,0,0] for a small offset, or 0 if you want none.
    activate_tool_compensation = False
    tool_compensation_force = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 5) Desired nullspace posture
    #    q_d_nullspace
    #    We'll pick something near the rest posture, or from the initial q.
    #    We'll fill it in after we read the initial q from MuJoCo.
    q_d_nullspace = None
    q_d_nullspace_initialized = False

    # 6) Possibly define a torque-rate saturation
    #    We'll keep it simple with a function
    delta_tau_max = 100.0  # limit how fast torque can change

    def saturate_torque_rate(tau_d_calc, tau_prev):
        """
        Matches the idea: 
          difference = tau_d_calc[i] - tau_prev[i]
          clamp difference to +/- delta_tau_max
          tau_d_saturated[i] = tau_prev[i] + difference
        """
        diff = tau_d_calc - tau_prev
        diff_clamped = jnp.clip(diff, -delta_tau_max, delta_tau_max)
        return tau_prev + diff_clamped

    # Create a viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # For demonstration, add a small frame to show x_des/quat_des
        frame_id = init_desired_pose_frame(viewer, 3)

        # 7) We want to initialize q_d_nullspace from the initial state
        mujoco.mj_forward(model, data)
        q_init = jnp.array(data.qpos[:7])  # first 7
        q_d_nullspace = q_init.copy()
        q_d_nullspace_initialized = True

        # Track the last commanded torque to saturate torque rate
        tau_prev = jnp.zeros(7)

        # Simulation loop
        render_period = 1.0 / render_fps
        sim_start = time.time()
        for step_i in range(steps):
            loop_start = time.time()

            #------------------------------------------
            # 1) Forward the model
            #------------------------------------------
            mujoco.mj_forward(model, data)

            #------------------------------------------
            # 2) Read current joint states, Jacobian
            #------------------------------------------
            q = jnp.array(data.qpos[:7])
            dq = jnp.array(data.qvel[:7])

            # body frame name
            body_name = "hand"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            x_curr = jnp.array(data.xpos[body_id])          # end-effector position
            rot_ee = jnp.array(data.xmat[body_id].reshape((3,3)))  # orientation matrix
            # get the orientation as a quaternion if needed
            # or we can just use rot_ee in the error logic
            quat_curr_mj = jnp.array(data.xquat[body_id])   # w, x, y, z

            # Jacobian
            jacp, jacr = get_body_jacobian(model, data, body_name)
            jacp = jacp[:, :7]
            jacr = jacr[:, :7]
            J = jnp.concatenate([jacp, jacr], axis=0)  # 6x7

            #------------------------------------------
            # 3) Compute task-space error
            #    position error + orientation error (C++ approach)
            #------------------------------------------
            e_pos = x_curr - x_des  # note the C++ code does (position - position_d_)
            # we can replicate that sign exactly:
            #   error(0..2) = position - position_d_ => we'll do e_pos = (position - x_des)
            e_pos = (x_curr - x_des)

            # orientation error
            # The Franka code: orientation = current, orientation_d = desired
            # error_quaternion = orientation.inverse() * orientation_d
            # e_ori = - R_curr * error_quaternion[xyz]
            # We'll replicate that
            R_curr = rot_ee
            e_ori = orientation_error(quat_curr_mj, quat_des, R_curr)

            # Combine
            error_6d = jnp.concatenate([e_pos, e_ori], axis=0)

            #------------------------------------------
            # 4) Task-space velocity
            #------------------------------------------
            v_pos = jacp @ dq
            v_ori = jacr @ dq
            v_6d  = jnp.concatenate([v_pos, v_ori], axis=0)

            #------------------------------------------
            # 5) Cartesian PD => F_ee_des
            #    (like -cartesian_stiffness_ * error - cartesian_damping_ * velocity)
            #------------------------------------------
            F_ee_des = - cartesian_stiffness @ error_6d - cartesian_damping @ v_6d

            #------------------------------------------
            # 6) tau_task = J^T * F_ee_des
            #------------------------------------------
            tau_task = J.T @ F_ee_des

            #------------------------------------------
            # 7) Nullspace torque
            #    tau_null = (I - J^T pinv(J^T)) * [K_ns (q_d - q) - 2 sqrt(K_ns) dq]
            #------------------------------------------
            # Pseudoinverse of J^T is (J^T)^+
            # but we have J 6x7 => J^T 7x6 => we invert that 7x6
            Jt_pinv = pseudo_inverse(J.T)  # shape ~ (6,7)
            # (I - J^T pinv(J^T)) => 7x7 - (7x6 @ 6x7) => 7x7
            proj = jnp.eye(7) - (J.T @ Jt_pinv)
            # Nullspace PD
            if q_d_nullspace_initialized:
                e_ns = (q_d_nullspace - q)
            else:
                e_ns = jnp.zeros_like(q)
            d_ns = dq
            # 2 * sqrt(K_ns)
            dn = 2.0 * jnp.sqrt(nullspace_stiffness)
            tau_null = proj @ (nullspace_stiffness * e_ns - dn * d_ns)

            #------------------------------------------
            # 8) Tool compensation 
            #    tau_tool = J^T * F_tool
            #    if "activate_tool_compensation_" is True
            #------------------------------------------
            if activate_tool_compensation:
                # Suppose we define tool_compensation_force as R^0_ee x something if needed
                # but in the C++ code it's in end-effector frame or base frame? 
                # Usually you'd want a force in end-effector frame, so you might map it
                # but the example code does:
                #   tau_tool << jacobian.transpose() * tool_compensation_force_;
                # that means tool_compensation_force_ is in the E.E. frame if jac= base? 
                # We'll just replicate that logic:
                tau_tool = J.T @ tool_compensation_force
            else:
                tau_tool = jnp.zeros(7)

            #------------------------------------------
            # 9) Coriolis + final torque
            #    tau_d = tau_task + tau_null + coriolis - tau_tool
            #------------------------------------------
            # But we don't have Franka's model_handle->getCoriolis() here. 
            # We can approximate with MuJoCo:
            #   We do mj_inverseDynamics to fill data.qfrc_bias with gravity/coriolis
            #   Then coriolis ~ data.qfrc_bias - mg? 
            #   Or we can just treat "bias" as gravity + Coriolis. 
            # For simplicity, let's treat qfrc_bias as gravity + Coriolis. 
            mujoco.mj_inverse(model, data)
            # data.qfrc_bias => "forces from gravity, Coriolis, and centrifugal"
            cor_grav = jnp.array(data.qfrc_bias[:7])

            tau_d = tau_task + tau_null + cor_grav - tau_tool

            #------------------------------------------
            # 10) Saturate torque rate
            #     tau_d = saturateTorqueRate(tau_d, tau_prev)
            #------------------------------------------
            tau_d = saturate_torque_rate(tau_d, tau_prev)
            tau_prev = tau_d

            #------------------------------------------
            # 11) Send to actuators
            #------------------------------------------
            data.ctrl[:7] = np.array(tau_d)

            # Step the simulation
            mujoco.mj_step(model, data)

            #------------------------------------------
            # Visualization & sleeping for real-time
            #------------------------------------------
            update_desired_pose_frame(viewer, frame_id, x_des, quat_des, length=0.1)
            viewer.sync()
            if not viewer.is_running():
                break

            elapsed = time.time() - loop_start
            if elapsed < render_period:
                time.sleep(render_period - elapsed)

        # End of simulation loop
        total = time.time() - sim_start
        print(f"Finished after {total:.2f}s of sim-time.")

###############################################################################
# Entry
###############################################################################

if __name__ == "__main__":
    xml_file = "models/mujoco_menagerie/franka_emika_panda/mjx_scene.xml"  # must have motor actuators for torque
    run_cartesian_pose_impedance_control(
        xml_path=xml_file,
        sim_time=10.0,
        render_fps=60.0,
        fixed_camera_id=None
    )
