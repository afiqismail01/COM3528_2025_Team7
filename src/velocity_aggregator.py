#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import TwistStamped

class VelocityAggregator:
    def __init__(self):
        rospy.init_node("velocity_aggregator", anonymous=True)
        self.pub = rospy.Publisher("/all_miro_velocities", TwistStamped, queue_size=10)

        miro_ids = ["miro01", "miro02", "miro03", "miro04", "miro05"]
        for robot in miro_ids:
            topic = f"/{robot}/velocity"
            rospy.Subscriber(topic, TwistStamped, self.forward_callback, callback_args=robot)

        rospy.loginfo("Velocity aggregator running...")
        rospy.spin()

    def forward_callback(self, msg, robot_id):
        self.pub.publish(msg)

if __name__ == "__main__":
    try:
        VelocityAggregator()
    except rospy.ROSInterruptException:
        pass
