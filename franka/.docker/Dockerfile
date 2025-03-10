FROM osrf/ros:noetic-desktop-full

##################
# parameters
##################

# Set IP of Franka control PC
ENV FCI_IP=172.16.0.2

##################
# libfranka build
##################

# Download and build the required franka libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libpoco-dev \
    libeigen3-dev

# Add Gazebo-related packages
RUN apt-get update && apt-get install -y \
ros-noetic-gazebo-ros-pkgs \
ros-noetic-gazebo-ros-control \
ros-noetic-joint-state-controller \
ros-noetic-effort-controllers \
ros-noetic-position-controllers

RUN git clone --recursive https://github.com/frankaemika/libfranka --branch 0.10.0 # only for FR3
WORKDIR /libfranka
RUN mkdir build
WORKDIR /libfranka/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
RUN cmake --build .

# Make a Debian package and install it
RUN cpack -G DEB
RUN dpkg -i libfranka*.deb

##################
# franka_ros build
##################

# Setup ROS catkin workspace
WORKDIR /catkin_ws
RUN mkdir src
# so we can use `source`
SHELL ["/bin/bash", "-c"]  
RUN source /opt/ros/noetic/setup.sh && catkin_init_workspace src

# Download and build
RUN git clone --recursive https://github.com/frankaemika/franka_ros src/franka_ros
RUN rosdep install --from-paths src --ignore-src --rosdistro noetic -y --skip-keys libfranka
RUN source /opt/ros/noetic/setup.sh && catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/libfranka/build

# WORKDIR /catkin_ws/src
# RUN git clone https://github.com/frankaemika/franka_ros_interface.git
# RUN git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git



# Add lines to the bashrc file that source ROS
RUN echo "source /ros_entrypoint.sh" >> /root/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc