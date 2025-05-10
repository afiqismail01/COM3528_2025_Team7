#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import TwistStamped

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.lib import wheel_speed2cmd_vel  # Python 2

class MiroClient:
    """
    Controls MiRo movement using /control/cmd_vel topic.
    """

    ##########################
    TICK = 0.02
    IS_MIROCODE = False  # Set to True if running in MiRoCODE
    ##########################

    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo
        if not self.IS_MIROCODE:
            rospy.init_node("circle_motion", anonymous=True)

        # Create a publisher for MiRo's movement commands
        miro_name = os.getenv("MIRO_ROBOT_NAME")
        topic_base_name = f"/{miro_name}/control/cmd_vel"
        self.vel_pub = rospy.Publisher(topic_base_name, TwistStamped, queue_size=10)

        # Give ROS time to start up properly
        rospy.sleep(2.0)

        # Move the head to the default pose
        self.reset_head_pose()

    def reset_head_pose(self):
        """Resets MiRo's head position (optional)."""
        pass  # Add head reset logic if needed

    def drive(self, speed_l=0.1, speed_r=0.3):  
        """
        Moves MiRo in a **circular path** by setting different left & right wheel speeds.
        """

        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Convert wheel speeds (m/sec) to command velocity (linear & angular)
        (dr, dtheta) = wheel_speed2cmd_vel([speed_l, speed_r])

        # Update the message with calculated movement values
        msg_cmd_vel.twist.linear.x = dr  # Forward speed
        msg_cmd_vel.twist.angular.z = dtheta  # Turning speed

        # Publish the message to MiRo
        self.vel_pub.publish(msg_cmd_vel)

    def loop(self):
        """
        Main control loop to continuously move MiRo in circles.
        """
        rospy.loginfo("MiRo moving in a circular motion...")
        
        rate = rospy.Rate(1 / self.TICK)  # Set loop rate
        
        while not rospy.core.is_shutdown():
            self.drive(speed_l=0.4, speed_r=0.35) # Adjust these values for different circular paths
            rate.sleep()

if __name__ == "__main__":
    main = MiroClient()
    main.loop()
