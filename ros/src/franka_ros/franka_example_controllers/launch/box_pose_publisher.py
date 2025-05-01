#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped


class BoxPosePublisher:
    def __init__(self):
        rospy.init_node("box_pose_publisher")

        # Get the model path from ROS parameter
        self.model_path = rospy.get_param("~model_path", "")
        if not self.model_path:
            rospy.logerr("No model path provided!")
            return

        # Wait for Gazebo services
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        # Publisher for box pose
        self.box_pose_pub = rospy.Publisher("/box/pose", PoseStamped, queue_size=10)

        # Subscribe to Gazebo model states
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_states_callback
        )

        # Spawn the box
        self.spawn_box()

    def spawn_box(self):
        # Read the box SDF file
        try:
            with open(self.model_path, "r") as f:
                box_sdf = f.read()
        except Exception as e:
            rospy.logerr(f"Failed to read SDF file: {e}")
            return

        # Initial pose for the box
        initial_pose = Pose()
        initial_pose.position = Point(0.4, 0.0, 0.03)  # Adjust position as needed
        initial_pose.orientation = Quaternion(0, 0, 0, 1)

        # Spawn the box model
        try:
            self.spawn_model("box", box_sdf, "", initial_pose, "world")
            rospy.loginfo("Box spawned successfully")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn box: {e}")

    def model_states_callback(self, msg):
        try:
            # Find the index of our box in the model_states
            box_idx = msg.name.index("box")

            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "world"
            pose_msg.pose = msg.pose[box_idx]

            # Publish the pose
            self.box_pose_pub.publish(pose_msg)

        except ValueError:
            pass  # Box not found in model_states


if __name__ == "__main__":
    try:
        box_publisher = BoxPosePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
