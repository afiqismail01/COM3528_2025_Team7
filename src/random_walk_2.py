#!/usr/bin/env python3
import os
import random
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
        rospy.loginfo("MiRo moving with sharper obstacle avoidance...")

        rate = rospy.Rate(1 / self.TICK)
        turning_counter = 0
        turning_steps = 60  # ~1.2 seconds at 0.02s per tick
        turn_dir = 1  # Can be randomised if you want later

        while not rospy.core.is_shutdown():
            if self.sonar_distance is not None:
                d = self.sonar_distance

                if turning_counter > 0:
                    rospy.logwarn(f"Turning sharply... {turning_counter} steps remaining")
                    self.drive(speed_l=0.2 * turn_dir, speed_r=-0.2 * turn_dir)  # Sharper turn
                    turning_counter -= 1

                elif d < 0.135:
                    rospy.logwarn(f"Obstacle too close ({d:.2f} m)! Initiating sharp turn.")
                    turning_counter = turning_steps
                    turn_dir = random.choice([-1, 1])  # Optional: random left or right
                    self.drive(speed_l=0.2 * turn_dir, speed_r=-0.2 * turn_dir)

                else:
                    self.drive(speed_l=0.2, speed_r=0.2)  # Go straight

            else:
                rospy.loginfo("Waiting for sonar data...")

            rate.sleep()



if __name__ == "__main__":
    main = MiroClient()
    main.loop()
