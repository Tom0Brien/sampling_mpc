import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

def euler_to_quat(rpy):
    """Convert Euler angles (roll, pitch, yaw) into a quaternion [w, x, y, z]."""
    roll, pitch, yaw = rpy
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
        [1 - 2*(y**2 + z**2),    2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),          1 - 2*(x**2 + z**2),2*(y*z - w*x)],
        [2*(x*z - w*y),          2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
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

def add_desired_pose_frame(viewer, base_id, x_des, quat_des, length=0.1):
    """Add line geoms to show the desired frame origin and axes."""
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
    return base_id