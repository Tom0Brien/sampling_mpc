#!/bin/bash

# Source ROS
source /opt/ros/noetic/setup.bash

# Build the whole workspace
cd /catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release