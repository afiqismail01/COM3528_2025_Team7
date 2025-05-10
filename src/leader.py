#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, Range

SAFE_DISTANCE = 0.135
TURN_DURATION = 0.6  # seconds for approx 180 turn
TICK = 0.1

def sonar_callback(msg):
    global sonar_distance
    sonar_distance = msg.range

if __name__ == "__main__":
    rospy.init_node("leader", anonymous=True)
    miro_name = os.getenv("MIRO_ROBOT_NAME", "miro01")
    topic_base_name = "/" + miro_name
    vel_pub = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)
    yaw_pub = rospy.Publisher(topic_base_name + "/yaw", JointState, queue_size=1)
    sonar_distance = None
    rospy.Subscriber(topic_base_name + "/sensors/sonar", Range, sonar_callback)
    rate = rospy.Rate(1 / TICK)
    yaw = 0.0

    while not rospy.is_shutdown():
        msg = TwistStamped()
        if sonar_distance is not None and sonar_distance < SAFE_DISTANCE:
            turn_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - turn_time < TURN_DURATION and not rospy.core.is_shutdown():
                msg.twist.linear.x = 0.0
                msg.twist.angular.z = 0.5
                vel_pub.publish(msg)
                yaw += msg.twist.angular.z * TICK
                js = JointState()
                js.position = [0.0, 0.0, yaw, 0.0]
                yaw_pub.publish(js)
                rospy.sleep(TICK)
            msg.twist.linear.x = 0.15
            msg.twist.angular.z = 0.0
        else:
            msg.twist.linear.x = 0.15
            msg.twist.angular.z = 0.0

        vel_pub.publish(msg)
        yaw += msg.twist.angular.z * TICK
        js = JointState()
        js.position = [0.0, 0.0, yaw, 0.0]
        yaw_pub.publish(js)
        rate.sleep()