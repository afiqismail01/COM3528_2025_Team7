#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, Range

SAFE_DISTANCE = 0.135
TURN_DURATION = 0.6
TICK = 0.1

class BoidsAlignment:
    def __init__(self):
        self.role = os.getenv("MIRO_ROBOT_ROLE", "follower")  # "leader" or "follower"
        self.miro_name = os.getenv("MIRO_ROBOT_NAME", "miro01")
        self.topic_base_name = "/" + self.miro_name

        self.vel_pub = rospy.Publisher(
            self.topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )

        if self.role == "leader":
            self.yaw_pub = rospy.Publisher(self.topic_base_name + "/yaw", JointState, queue_size=1)
            self.sonar_distance = None
            rospy.Subscriber(self.topic_base_name + "/sensors/sonar", Range, self.sonar_callback)
            self.yaw = 0.0
        else:
            self.target_name = os.getenv("LEADER_NAME", "miro01")
            self.target_vel = [0.0, 0.0]
            rospy.Subscriber(
                f"/{self.target_name}/control/cmd_vel", TwistStamped, self.target_vel_callback
            )

    def sonar_callback(self, msg):
        self.sonar_distance = msg.range

    def target_vel_callback(self, msg):
        self.target_vel = [msg.twist.linear.x, msg.twist.angular.z]

    def drive(self, linear, angular):
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.twist.linear.x = linear
        msg_cmd_vel.twist.angular.z = angular
        self.vel_pub.publish(msg_cmd_vel)

    def run_leader(self):
        rate = rospy.Rate(1 / TICK)
        yaw = 0.0
        while not rospy.is_shutdown():
            msg = TwistStamped()
            if self.sonar_distance is not None and self.sonar_distance < SAFE_DISTANCE:
                turn_time = rospy.Time.now().to_sec()
                while rospy.Time.now().to_sec() - turn_time < TURN_DURATION and not rospy.core.is_shutdown():
                    msg.twist.linear.x = 0.0
                    msg.twist.angular.z = 0.5
                    self.vel_pub.publish(msg)
                    yaw += msg.twist.angular.z * TICK
                    js = JointState()
                    js.position = [0.0, 0.0, yaw, 0.0]
                    self.yaw_pub.publish(js)
                    rospy.sleep(TICK)
                msg.twist.linear.x = 0.15
                msg.twist.angular.z = 0.0
            else:
                msg.twist.linear.x = 0.15
                msg.twist.angular.z = 0.0

            self.vel_pub.publish(msg)
            yaw += msg.twist.angular.z * TICK
            js = JointState()
            js.position = [0.0, 0.0, yaw, 0.0]
            self.yaw_pub.publish(js)
            rate.sleep()

    def run_follower(self):
        rate = rospy.Rate(1 / TICK)
        while not rospy.is_shutdown():
            self.drive(self.target_vel[0], self.target_vel[1])
            rate.sleep()

    def run(self):
        if self.role == "leader":
            self.run_leader()
        else:
            self.run_follower()

if __name__ == "__main__":
    rospy.init_node("boids_alignment", anonymous=True)
    client = BoidsAlignment()
    client.run()