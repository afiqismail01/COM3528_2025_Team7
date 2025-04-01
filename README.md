# com3528_2025_Team7

Basic setup for MiRo project.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)

## Setup

1. Clone the project repository into ROS workspace:

    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/afiqismail01/COM3528_2025_Team7.git
    ```

2. Build workspace:

    ```bash
    cd ~/catkin_ws
    catkin build com3528_2025_Team7
    source devel/setup.bash
    ```

    If an error occurs during the build process, clean the workspace and try again:

    ```bash
    catkin clean
    ```
## Usage
All created development python codes should be placed in `src` directory.

1. Make the script in the `src` folder executable:

    ```bash
    chmod +x ~/catkin_ws/src/COM3528_2025_Team7/src/<script_name>.py
    ```

2. Start ROS:

    ```bash
    roslaunch com3528_2025_Team7 empty_world.launch
    ```

## Development Helper

1. Target MiRo before running `rosrun` command for assigning the code to a particular MiRo

    ```bash
    // MIRO_ROBOT_NAME=miro01
    export MIRO_ROBOT_NAME=<miro_robot_name>
    ```

2. Rosrun:

    ```bash
    rosrun com3528_2025_Team7 <script_name>.py
    ```