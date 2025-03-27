# com3528_2025_Team7

Basic setup for MiRo project.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)

## Setup

1. Clone the project repository into ROS workspace:

    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/EIliott/COM3528_2025_Team7.git
    ```

2. Build workspace:

    ```bash
    cd ~/catkin_ws
    catkin build
    source devel/setup.bash
    ```
## Usage

1. Start ROS:

    ```bash
    roslaunch com3528_2025_Team7 empty_world.launch
    ```
