#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Range

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
            rospy.init_node("smart_forward", anonymous=True)

        # Create a publisher for MiRo's movement commands
        miro_name = os.getenv("MIRO_ROBOT_NAME")
        topic_base_name = f"/{miro_name}/control/cmd_vel"
        self.vel_pub = rospy.Publisher(topic_base_name, TwistStamped, queue_size=10)

        # Subscribe to sonar sensor
        sonar_topic = f"/{miro_name}/sensors/sonar"
        self.sonar_distance = None
        rospy.Subscriber(sonar_topic, Range, self.sonar_callback)

        # Give ROS time to start up properly
        rospy.sleep(2.0)

        # Move the head to the default pose
        self.reset_head_pose()

    def sonar_callback(self, msg):
        self.sonar_distance = msg.range
        distance_cm = self.sonar_distance
        rospy.loginfo_throttle(1.0, f"[Sonar] Distance: {distance_cm} cm")


    def reset_head_pose(self):
        """Resets MiRo's head position (optional)."""
        pass  # Add head reset logic if needed

    def drive(self, speed_l=0.2, speed_r=0.2):
        """
        Drives MiRo with given wheel speeds.
        """
        msg_cmd_vel = TwistStamped()
        (dr, dtheta) = wheel_speed2cmd_vel([speed_l, speed_r])
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta
        self.vel_pub.publish(msg_cmd_vel)

    def loop(self):
        rospy.loginfo("MiRo moving with smarter obstacle avoidance...")

        rate = rospy.Rate(1 / self.TICK)
        turning_counter = 0
        turning_steps = 30  # 30 x 0.02s = 0.6 seconds of turning

        while not rospy.core.is_shutdown():
            if self.sonar_distance is not None:
                distance_cm = self.sonar_distance
                # rospy.logwarn(f"Obstacle too close ({distance_cm} cm)! Initiating turn.")

            else:
                rospy.loginfo("Waiting for sonar data...")

            rospy.sleep(2.0)


if __name__ == "__main__":
    main = MiroClient()
    main.loop()
