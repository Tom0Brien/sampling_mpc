FROM osrf/ros:noetic-desktop-full

##################
# parameters
##################

# Set IP of Franka control PC
ENV FCI_IP=172.16.0.2

##################
# Install dependencies
##################

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libpoco-dev \
    libeigen3-dev \
    python3-pip \
    ros-noetic-rosbridge-server \
    ros-noetic-tf2-web-republisher \
    ros-noetic-combined-robot-hw \
    ros-noetic-combined-robot-hw-tests \
    ros-noetic-hardware-interface \
    ros-noetic-controller-manager \
    ros-noetic-controller-interface \
    ros-noetic-joint-limits-interface \
    ros-noetic-transmission-interface \
    ros-noetic-boost-sml

# Add Gazebo-related packages
RUN apt-get install -y \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-controller \
    ros-noetic-effort-controllers \
    ros-noetic-position-controllers \
    ros-noetic-joint-trajectory-controller \
    ros-noetic-moveit \
    ros-noetic-moveit-ros-move-group \
    ros-noetic-moveit-fake-controller-manager \
    ros-noetic-moveit-kinematics \
    ros-noetic-moveit-planners-ompl \
    ros-noetic-moveit-ros-visualization

# Install Kinova-specific dependencies
RUN apt-get install -y \
    ros-noetic-rqt-gui \
    ros-noetic-rqt-gui-py \
    libusb-1.0-0-dev

# Install and configure Conan for Kortex API
RUN pip3 install conan==1.59
RUN conan config set general.revisions_enabled=1
RUN conan profile new default --detect > /dev/null
RUN conan profile update settings.compiler.libcxx=libstdc++11 default
# Fix for Conan SSL certificate issue
RUN conan config install https://github.com/conan-io/conanclientcert.git

# Franka-specific build
RUN git clone --recursive https://github.com/frankaemika/libfranka --branch 0.10.0 # only for FR3
WORKDIR /libfranka
RUN mkdir build
WORKDIR /libfranka/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
RUN cmake --build .
# Make a Debian package and install it
RUN cpack -G DEB
RUN dpkg -i libfranka*.deb

# Setup ROS catkin workspace
WORKDIR /catkin_ws
RUN mkdir src
# so we can use `source`
SHELL ["/bin/bash", "-c"]  
RUN source /opt/ros/noetic/setup.sh && catkin_init_workspace src

##################
# Project setup
##################

# Add lines to the bashrc file that source ROS
RUN echo "source /ros_entrypoint.sh" >> /root/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

# Add lines to support UV development workflow
RUN echo 'export PATH="/app/.venv/bin:$PATH"' >> /root/.bashrc