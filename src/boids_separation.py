#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import Range
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
    IS_MIROCODE = False  # Set to True if running in MiRoCODE
    ##########################

    def __init__(self):
        rospy.init_node("movement_publisher")
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        
        # Create a publisher for MiRo's movement commands
        self.vel_pub = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=10)

        # ODOM subscriber
        self.fov = None                 # the field of view that is used by the sensor
        self.min_range = None           # the minimum range set to be detected by the sensor
        self.max_range = None           # the maximum range set to be detected by the sensor
        self.range = None               # distance to the object within the set boundaries of the sensor
        self.subscriber = rospy.Subscriber(topic_base_name + "/sensors/sonar", Range, self.callback)

    def callback(self, data):
        self.fov = data.field_of_view
        self.min_range = data.min_range
        self.max_range = data.max_range
        self.range = data.range

    def drive(self, linear = 0.0, angular = 0.0):  
        """
        Moves MiRo in a **straight line** by setting the same speed for both wheels.
        """
        vel_cmd = TwistStamped()
        # explanation of the messages in the document
        # message variable to move forward is done by linear.x
        vel_cmd.twist.linear.x = linear
        # message variable to turn is done by angular.z
        vel_cmd.twist.angular.z = angular
        self.vel_pub.publish(vel_cmd)

    def loop(self):
        """
        Main control loop.
        """
        rospy.loginfo("MiRo action starts...")
        
        while not rospy.is_shutdown():
            # Makes MiRo move in circles
            self.drive(linear=1.0, angular=0.0)
            print("Range: ", self.range)
            if self.range is not None and self.range < 0.3:
                self.drive(linear=0.0, angular=0.0)
                rospy.loginfo("Obstacle detected. Stopping MiRo.")
                rospy.signal_shutdown("Shutting down due to obstacle detection.")
            rospy.sleep(0.02)

if __name__ == "__main__":
    main = MiroClient()
    main.loop()
