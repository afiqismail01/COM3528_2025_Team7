#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import TwistStamped
import random

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.lib import wheel_speed2cmd_vel  # Python 2

class MiroClient:

    def __init__(self):
        rospy.init_node("random_walk_publisher")
        # Create a publisher for MiRo's movement commands
        miro_name = os.getenv("MIRO_ROBOT_NAME", "miro01")
        topic_base_name = f"/{miro_name}/control/cmd_vel"
        self.vel_pub = rospy.Publisher(topic_base_name, TwistStamped, queue_size=10)
        # Clean-up
        rospy.on_shutdown(self.shutdown_hook)

    # linear is to move straight and angular is to turn
    def set_move_cmd(self, linear = 0.0, angular = 0.0):
        vel_cmd = TwistStamped()
        # explanation of the messages in the document
        # message variable to move forward is done by linear.x
        vel_cmd.twist.linear.x = linear
        # message variable to turn is done by angular.z
        vel_cmd.twist.angular.z = angular
        self.vel_pub.publish(vel_cmd)

    def shutdown_hook(self):
        # Stop moving
        self.set_move_cmd()

if __name__ == "__main__":
    main = MiroClient()

    rospy.loginfo("MiRo action starts...")
    while not rospy.is_shutdown():
        # Makes MiRo moves in random directions
        linear_speed = round(random.uniform(0, 2.0), 2)
        angular_speed = round(random.uniform(0, 2.0), 2)
        rospy.loginfo(f"Linear speed: {linear_speed}, Angular speed: {angular_speed}")
        main.set_move_cmd(linear=linear_speed, angular=angular_speed)
        rospy.sleep(0.02)
