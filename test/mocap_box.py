"""
Simple script to continuously stream 6DOF data for a specific body ('box')
from QTM
"""

import asyncio
import xml.etree.ElementTree as ET
import qtm_rt
import numpy as np


def create_body_index(xml_string):
    """Extract a name to index dictionary from 6dof settings xml"""
    xml = ET.fromstring(xml_string)
    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index
    return body_to_index


def on_packet(packet, body_index):
    """Callback function that is called every time a data packet arrives from QTM"""
    info, bodies = packet.get_6d()

    # Get the box data using its index
    if "box" in body_index:
        box_index = body_index["box"]
        position, rotation = bodies[box_index]

        # Convert position to numpy array
        position_array = np.array([position.x, position.y, position.z])
        # Convert to meters
        position_array = position_array / 1000

        # Convert rotation matrix (9 values) to 3x3 numpy array
        rotation_array = np.array(rotation.matrix).reshape(3, 3)

        print(f"Frame: {packet.framenumber}")
        print(f"Position: {position_array}")
        print(f"Rotation Matrix:\n{rotation_array}")
        print("-" * 50)


async def main():
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
    callback = lambda packet: on_packet(packet, body_index)

    # Stream frames continuously
    await connection.stream_frames(components=["6d"], on_packet=callback)


if __name__ == "__main__":
    # Run forever
    asyncio.ensure_future(main())
    asyncio.get_event_loop().run_forever()
