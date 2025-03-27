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
    catkin build com3528_2025_Team7
    source devel/setup.bash
    ```

    If an error occurs during the build process, clean the workspace and try again:

    ```bash
    catkin clean
    ```
## Usage

1. Make the script in the `src` folder executable:

    ```bash
    chmod +x ~/catkin_ws/src/COM3528_2025_Team7/src/<script_name>.py
    ```

2. Start ROS:

    ```bash
    roslaunch com3528_2025_Team7 empty_world.launch
    ```



