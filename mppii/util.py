import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import mujoco.mjx as mjx


M_PI_4 = 0.7853981633974483
M_PI_2 = 1.5707963267948966


def eul_to_quat(rpy):
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
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])


def quat_flip_if_needed(q_curr, q_des):
    """Flip q_curr sign if needed to avoid discontinuity."""
    return jnp.where(jnp.dot(q_des, q_curr) < 0.0, -q_curr, q_curr)


def orientation_error(q_curr, q_des, R_world_ee):
    """Compute orientation error in base frame."""
    # Flip quaternion if dot product is negative
    q_curr = quat_flip_if_needed(q_curr, q_des)

    # Compute quaternion inverse
    w, x, y, z = q_curr
    q_inv = jnp.array([w, -x, -y, -z])

    # Difference quaternion (q_inv * q_des)
    q_err = quat_mul(q_inv, q_des)

    # Extract imaginary components (x,y,z) of error quaternion
    e_local = q_err[1:]

    # Transform to base frame (only negate once)
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
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def quat_to_eul(q):
    """Convert a unit quaternion [w,x,y,z] to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = jnp.arcsin(2 * (w * y - z * x))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return jnp.array([roll, pitch, yaw])


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
            rgba=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        viewer.user_scn.ngeom += 1
    return base_id


def update_desired_pose_frame(viewer, base_id, x_des, quat_des, length=0.1):
    """Update line geoms to show the desired frame origin and axes."""
    R = quat_to_mat(quat_des)
    axes = {
        0: (jnp.array([1, 0, 0]), np.array([1, 0, 0, 1])),
        1: (jnp.array([0, 1, 0]), np.array([0, 1, 0, 1])),
        2: (jnp.array([0, 0, 1]), np.array([0, 0, 1, 1])),
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
            rgba=color,
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start_pt, end_pt)


def add_desired_pose_frame(viewer, base_id, x_des, quat_des, length=0.1):
    """Add line geoms to show the desired frame origin and axes."""
    R = quat_to_mat(quat_des)
    axes = {
        0: (jnp.array([1, 0, 0]), np.array([1, 0, 0, 1])),
        1: (jnp.array([0, 1, 0]), np.array([0, 1, 0, 1])),
        2: (jnp.array([0, 0, 1]), np.array([0, 0, 1, 1])),
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
            rgba=color,
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start_pt, end_pt)
    return base_id


def mat_to_quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    JAX-compatible implementation using jnp.where instead of conditionals.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    # Case 1: trace > 0
    S1 = jnp.sqrt(trace + 1.0) * 2
    w1 = 0.25 * S1
    x1 = (R[2, 1] - R[1, 2]) / S1
    y1 = (R[0, 2] - R[2, 0]) / S1
    z1 = (R[1, 0] - R[0, 1]) / S1

    # Case 2: R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]
    S2 = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
    w2 = (R[2, 1] - R[1, 2]) / S2
    x2 = 0.25 * S2
    y2 = (R[0, 1] + R[1, 0]) / S2
    z2 = (R[0, 2] + R[2, 0]) / S2

    # Case 3: R[1, 1] > R[2, 2]
    S3 = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
    w3 = (R[0, 2] - R[2, 0]) / S3
    x3 = (R[0, 1] + R[1, 0]) / S3
    y3 = 0.25 * S3
    z3 = (R[1, 2] + R[2, 1]) / S3

    # Case 4: default case
    S4 = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    w4 = (R[1, 0] - R[0, 1]) / S4
    x4 = (R[0, 2] + R[2, 0]) / S4
    y4 = (R[1, 2] + R[2, 1]) / S4
    z4 = 0.25 * S4

    # Condition masks
    cond1 = trace > 0
    cond2 = jnp.logical_and(
        jnp.logical_not(cond1), jnp.logical_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2])
    )
    cond3 = jnp.logical_and(
        jnp.logical_not(cond1),
        jnp.logical_and(jnp.logical_not(cond2), R[1, 1] > R[2, 2]),
    )

    # Select w based on conditions
    w = jnp.where(cond1, w1, jnp.where(cond2, w2, jnp.where(cond3, w3, w4)))

    # Select x based on conditions
    x = jnp.where(cond1, x1, jnp.where(cond2, x2, jnp.where(cond3, x3, x4)))

    # Select y based on conditions
    y = jnp.where(cond1, y1, jnp.where(cond2, y2, jnp.where(cond3, y3, y4)))

    # Select z based on conditions
    z = jnp.where(cond1, z1, jnp.where(cond2, z2, jnp.where(cond3, z3, z4)))

    return jnp.array([w, x, y, z])


def mat_to_rpy(R):
    """Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw)."""
    # Handle singularity case (gimbal lock) when R[2,0] is close to +/-1
    # pitch = -arcsin(R[2,0])
    pitch = -jnp.arcsin(jnp.clip(R[2, 0], -1.0, 1.0))

    # Compute roll and yaw based on pitch
    # Use epsilon to avoid division by zero in the singularity case
    epsilon = 1e-10
    cos_pitch = jnp.cos(pitch)

    # roll = atan2(R[2,1]/cos(pitch), R[2,2]/cos(pitch))
    roll = jnp.arctan2(R[2, 1] / (cos_pitch + epsilon), R[2, 2] / (cos_pitch + epsilon))

    # yaw = atan2(R[1,0]/cos(pitch), R[0,0]/cos(pitch))
    yaw = jnp.arctan2(R[1, 0] / (cos_pitch + epsilon), R[0, 0] / (cos_pitch + epsilon))

    return jnp.array([roll, pitch, yaw])
