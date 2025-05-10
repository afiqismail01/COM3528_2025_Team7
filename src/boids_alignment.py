#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState

class SimpleFollow:
    TICK = 0.1

    def __init__(self):
        rospy.init_node("simple_follow", anonymous=True)
        self.miro_name = os.getenv("MIRO_ROBOT_NAME", "miro02")
        topic_base_name = "/" + self.miro_name

        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )

        self.target_vel = [0.0, 0.0]
        self.target_name = "miro01"
        rospy.Subscriber(
            f"/{self.target_name}/control/cmd_vel", TwistStamped, self.target_vel_callback
        )

    def target_vel_callback(self, msg):
        self.target_vel = [msg.twist.linear.x, msg.twist.angular.z]

    def drive(self, linear, angular):
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.twist.linear.x = linear
        msg_cmd_vel.twist.angular.z = angular
        self.vel_pub.publish(msg_cmd_vel)

    def loop(self):
        rate = rospy.Rate(1 / self.TICK)
        while not rospy.is_shutdown():
            self.drive(self.target_vel[0], self.target_vel[1])
            rate.sleep()

if __name__ == "__main__":
    client = SimpleFollow()
    client.loop()