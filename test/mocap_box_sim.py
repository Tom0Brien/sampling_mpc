"""
Simple script to continuously stream 6DOF data for a specific body ('box')
from QTM and visualize it in MuJoCo
"""

import asyncio
import xml.etree.ElementTree as ET
import qtm_rt
import numpy as np
import mujoco
import mujoco.viewer
import threading
from hydrax.util import mat_to_quat


class BoxState:
    """Thread-safe box state container"""

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.rotation = np.eye(3, dtype=np.float32)
        self.lock = threading.Lock()
        self.viewer_running = True

    def update_state(self, position, rotation):
        with self.lock:
            self.position[:] = position
            self.rotation[:] = rotation


def run_viewer(model_path: str, box_state: BoxState):
    """Run the MuJoCo viewer in a separate thread"""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with box_state.lock:
                # Update box position and orientation
                data.qpos[:3] = box_state.position
                data.qpos[3:7] = mat_to_quat(box_state.rotation)

            # Update the viewer
            mujoco.mj_step(model, data)
            viewer.sync()

    box_state.viewer_running = False


def create_body_index(xml_string):
    """Extract a name to index dictionary from 6dof settings xml"""
    xml = ET.fromstring(xml_string)
    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index
    return body_to_index


def on_packet(packet, body_index, box_state):
    """Callback function that is called every time a data packet arrives from QTM"""
    info, bodies = packet.get_6d()

    # Get the box data using its index
    if "box" in body_index:
        box_index = body_index["box"]
        position, rotation = bodies[box_index]

        # Convert position to numpy array and to meters
        position_array = np.array([position.x, position.y, position.z]) / 1000.0

        # Offset the position to the center of the box
        position_array -= np.array([0.0, 0.0, 0.077])

        # Convert rotation matrix (9 values) to 3x3 numpy array
        rotation_array = np.array(rotation.matrix).reshape(3, 3)

        # Only update the state if non-nan
        if not np.isnan(position_array).any() and not np.isnan(rotation_array).any():
            # Update shared state
            box_state.update_state(position_array, rotation_array)

        print(f"Frame: {packet.framenumber}")
        print(f"Position: {position_array}")
        print(f"Rotation Matrix:\n{rotation_array}")
        print("-" * 50)


async def main():
    # Initialize state container
    box_state = BoxState()

    # Start the MuJoCo viewer thread
    viewer_thread = threading.Thread(
        target=run_viewer,
        args=("hydrax/models/pushbox/scene.xml", box_state),
        daemon=True,
    )
    viewer_thread.start()

    # Connect to QTM
    connection = await qtm_rt.connect("192.168.1.10")
    if connection is None:
        print("Failed to connect to QTM")
        return

    # Get 6DOF settings from QTM
    xml_string = await connection.get_parameters(parameters=["6d"])
    body_index = create_body_index(xml_string)

    if "box" not in body_index:
        print("No body named 'box' found in the tracking data")
        return

    # Create callback with body_index included
    callback = lambda packet: on_packet(packet, body_index, box_state)

    # Stream frames continuously
    try:
        await connection.stream_frames(components=["6d"], on_packet=callback)
    except Exception as e:
        print(f"Error in QTM streaming: {e}")


if __name__ == "__main__":
    try:
        asyncio.ensure_future(main())
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
