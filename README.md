# COM3528_2025_Team7

**Emergent Swarm Intelligence in Decentralized MiRo Agents**
The project uses Boid Algorithm to simulate emergent group behaviour by layering Separation, Alignment, and Cohesion principles. The outcome of the simulation will form a group of MiRo moving in a same direction with no central control.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)

## Simulation Setup

1. Clone the project repository into ROS workspace:

    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/afiqismail01/COM3528_2025_Team7.git
    ```

2. Build workspace:

    ```bash
    cd ~/catkin_ws
    catkin build com3528_2025_team7
    source devel/setup.bash
    ```

    If an error occurs during the build process, clean the workspace and try again:

    ```bash
    catkin clean
    ```
## Usage
All created development python codes should be placed in `src` directory.

1. Make the boid_main_v3 script in the `src` folder executable:

    ```bash
    chmod +x ~/catkin_ws/src/com3528_2025_team7/src/boid_main_v3.py
    ```

2. Start ROS launcher to see the Boid Algorithm simulation using 5 MiRos:

    ```bash
    roslaunch com3528_2025_team7 sim_football.launch
    ```

## Optional Development Helper

1. Target MiRo before running `rosrun` command for assigning the code to a particular MiRo

    ```bash
    // MIRO_ROBOT_NAME=miro01
    export MIRO_ROBOT_NAME=<miro_robot_name>
    ```

2. Rosrun:

    ```bash
    rosrun com3528_2025_team7 boid_main_v3.py

    ```
