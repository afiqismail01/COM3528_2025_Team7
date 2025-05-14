#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes MiRo look for a blue ball and follow it

The code was tested for Python 2 and 3
For Python 2 you might need to change the shebang line to
#!/usr/bin/env python
"""

# Imports
import os
import random
import subprocess
from math import radians  # This is used to reset the head pose
import numpy as np  # Numerical Analysis library
import cv2  # Computer Vision library

import rospy  # ROS Python interface
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message
from sensor_msgs.msg import Range

import miro2 as miro  # Import MiRo Developer Kit library

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2

class MiRoClient:
    """
    Script settings below
    """
    ##########################
    TICK = 0.02  # This is the update interval for the main control loop in secs
    CAM_FREQ = 1  # Number of ticks before camera gets a new frame, increase in case of network lag
    DEBUG = False # Set to True to enable debug views of the cameras
    IS_MIROCODE = False  # Set to True if running in MiRoCODE

    # settings
    SAFE_DISTANCE = 0.135  # meters (adjust as needed)
    TURNING_FACTOR = 2  # Adjust this value to control the turning speed
    BASE_SPEED = 0.2  # Base speed for the robot
    FOLLOW_SPEED = 0.6  # Speed for following MiRo
    ROTATION_SPEED = 0.1  # Speed for rotating MiRo
    TURN_DURATION = 0.6  # seconds for approx 180 turn
    FOLLOW_STOP_LIMIT = 150  # Number of frames before triggering escape
    FACE_DETECTION_COOLDOWN = random.uniform(50.0, 200.0)   # seconds to ignore face detection
    ALIGNMENT_TIMEOUT = 4.0  # seconds to wait for alignment before switching to follow mode
    ##########################
    """
    End of script settings
    """

    def drive(self, speed_l=0.1, speed_r=0.1):
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        msg_cmd_vel = TwistStamped()
        wheel_speed = [speed_l, speed_r]
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        msg_cmd_vel.twist.linear.x = dr # Linear speed
        msg_cmd_vel.twist.angular.z = dtheta # Angular speed

        self.vel_pub.publish(msg_cmd_vel) # Publish the command

    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store image as class attribute for further use
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            # Raise the flag: A new frame is available for processing
            self.new_frame[index] = True
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass

    def detect_miro(self, frame_rgb, index=0, debug=True):
        if frame_rgb is None:
            return None

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        # --- GREY Range PERFECT ---
        lower_grey = np.array([0, 0, 30])
        upper_grey = np.array([180, 60, 160])
        mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)

        # --- WHITE Range ---
        lower_white = np.array([0, 0, 190])
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # --- Combine both masks ---
        mask = cv2.bitwise_or(mask_grey, mask_white)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Get largest white blob
        MIN_AREA = 800  # Adjust based on resolution — 800+ is good for 512x512
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

        if not filtered_contours:
            return None

        largest = max(filtered_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        radius = int(cv2.contourArea(largest) ** 0.5)

        if debug:
            cv2.drawContours(frame_rgb, [largest], -1, (0, 255, 0), 2)
            cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 255), -1)
            cv2.imshow(f"White Body Detection - Cam {index}", frame_rgb)
            cv2.waitKey(1)

        # Normalize
        h, w = frame_rgb.shape[:2]
        norm_x = (cx - w / 2) / w
        norm_y = (cy - h / 2) / w * -1.0
        norm_r = radius / w

        return [norm_x, norm_y, norm_r]

    def look_for_miro(self):
        """
        [1 of 3] Roam forward, and bounce off obstacles by turning 180°.
        """
        if self.just_switched:
            rospy.loginfo(f"[{self.miro_name}] wandering")
            self.just_switched = False

        # Check each camera for another MiRo
        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.target_miro[index] = self.detect_miro(image, index)

        if not self.target_miro[0] and not self.target_miro[1]:
            # No MiRo detected
            if self.sonar_distance is not None:
                d = self.sonar_distance
                if d < self.SAFE_DISTANCE:
                    # rospy.loginfo(f"[BOUNCE] Obstacle too close.")
                    # Turn in place: full spin, then move forward again
                    start_time = rospy.Time.now().to_sec()
                    while rospy.Time.now().to_sec() - start_time < self.TURN_DURATION and not rospy.core.is_shutdown():
                        self.drive(speed_l=self.TURNING_FACTOR, speed_r=-self.TURNING_FACTOR)
                        rospy.sleep(self.TICK)
                    # After turning, move forward
                    self.drive(speed_l=self.BASE_SPEED, speed_r=self.BASE_SPEED)
                else:
                    # Nothing ahead, just move
                    self.drive(speed_l=self.BASE_SPEED, speed_r=self.BASE_SPEED)
            else:
                rospy.loginfo("Waiting for sonar data...")
        else:
            # MiRo found!
            self.status_code = 2
            self.just_switched = True    
    
    def steer_away(self):
        """
        [2 of 3] If the ball is too close, rotate MiRo away from it.
        """
        if self.just_switched:
            self.just_switched = False

        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.target_miro[index] = self.detect_miro(image, index)

        # Proximity thresholds
        too_close_left = 0.1
        too_close_right = 0.1
        # rospy.loginfo(f"Left distance: {self.target_miro[0][2] if self.target_miro[0] else 'N/A'}, Right distance: {self.target_miro[1][2] if self.target_miro[1] else 'N/A'}")

        # --- Steer away only ---
        if self.target_miro[0] and self.target_miro[0][2] > too_close_left:
            # print("Ball too close on left — turning right to avoid")
            self.drive(self.ROTATION_SPEED, -self.ROTATION_SPEED)
            return

        elif self.target_miro[1] and self.target_miro[1][2] > too_close_right:
            # print("Ball too close on right — turning left to avoid")
            self.drive(-self.ROTATION_SPEED, self.ROTATION_SPEED)
            return

        # If ball is not too close (or not visible), proceed to next action
        # print("Ball is at a safe distance — switching to follow.")
        self.status_code = 3
        self.just_switched = True


    def follow(self):
        """
        [3 of 3] Once MiRO is in position, this function should drive the MiRo
        forward until it kicks the ball!
        """
        sonar_close = self.sonar_distance is not None and self.sonar_distance < self.SAFE_DISTANCE

        if self.just_switched:
            print("MiRo is kicking the ball!", sonar_close)
            self.just_switched = False

        if not sonar_close:
            self.drive(self.BASE_SPEED, self.BASE_SPEED)
        else:
            self.status_code = 0  # Back to the default state after the kick
            self.just_switched = True


    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo, if needed
        if not self.IS_MIROCODE:
            rospy.init_node("boid_main", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)
        # Initialise CV Bridge
        self.image_converter = CvBridge()
        self.miro_name = os.getenv('MIRO_ROBOT_NAME')
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        # Create two new subscribers to receive camera images with attached callbacks
        self.sub_caml = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_base_name + "/sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )
        # Subscribe to sonar sensor
        sonar_topic = f"{topic_base_name}/sensors/sonar"
        self.sonar_distance = None
        rospy.Subscriber(sonar_topic, Range, self.sonar_callback)

        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )
        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of ball's x, y, and r values for each camera
        self.target_miro = [None, None]
        # Set the default frame width (gets updated on receiving an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True

    def sonar_callback(self, msg):
        self.sonar_distance = msg.range
        distance_cm = self.sonar_distance
        # rospy.loginfo_throttle(1.0, f"[Sonar] Distance: {distance_cm} cm")

    def loop(self):
        """
        Main control loop
        """
        print("MiRo boids-separation, press CTRL+C to halt...")
        self.counter = 0
        self.status_code = 0
        self.status_start_time = rospy.Time.now().to_sec()

        while not rospy.core.is_shutdown():
            now = rospy.Time.now().to_sec()
            # Step 1. Find Miro
            if self.status_code == 1:
                # Every once in a while, look for miro
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_miro()

            # Step 2. Align away from Miro
            elif self.status_code == 2:
                if now - self.status_start_time > 1.0:
                    self.status_code = 3
                    self.status_start_time = now
                else:
                    self.steer_away()
                    if self.status_code == 3:
                        self.status_start_time = now

            # Step 3. Follow!
            elif self.status_code == 3:
                if now - self.status_start_time > 0.5:
                    self.status_code = 0
                    self.status_start_time = now
                else:
                    self.follow()
                    if self.status_code == 0:
                        self.status_start_time = now

            # Fall back
            else:
                self.status_code = 1
                self.status_start_time = now

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)


# This condition fires when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop