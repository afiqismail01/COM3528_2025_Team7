#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose2D

class StateAggregator:
    def __init__(self):
        rospy.init_node("state_aggregator", anonymous=True)
        self.pub = rospy.Publisher("/all_miro_states", Pose2D, queue_size=10)

        # Add subscribers for each MiRo agent
        miro_ids = ["miro01", "miro02", "miro03", "miro04", "miro05"]
        for robot in miro_ids:
            topic = f"/{robot}/state"
            rospy.Subscriber(topic, Pose2D, self.forward_callback, callback_args=robot)

        rospy.loginfo("Aggregator running...")
        rospy.spin()

    def forward_callback(self, msg, robot_id):
        # rospy.loginfo(f"Forwarding state from {robot_id}: x={msg.x}, y={msg.y}")
        self.pub.publish(msg)

if __name__ == "__main__":
    try:
        StateAggregator()
    except rospy.ROSInterruptException:
        pass
