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
    SLOW = 0.1  # Radial speed when turning on the spot (rad/s)
    FAST = 0.3  # Linear speed when following the ball (m/s)
    DEBUG = False # Set to True to enable debug views of the cameras
    TRANSLATION_ONLY = False # Whether to rotate only
    IS_MIROCODE = False  # Set to True if running in MiRoCODE

    # settings
    SAFE_DISTANCE = 0.135  # meters (adjust as needed)
    TURNING_FACTOR = 2  # Adjust this value to control the turning speed
    BASE_SPEED = 0.2  # Base speed for the robot
    TURN_DURATION = 0.6  # seconds for approx 180 turn
    FOLLOW_STOP_LIMIT = 70  # Number of frames before triggering escape
    FACE_DETECTION_COOLDOWN = 4.0  # seconds to ignore face detection
    ALIGNMENT_TIMEOUT = 4.0  # seconds to wait for alignment before switching to follow mode

    # formatting order
    PREPROCESSING_ORDER = ["edge", "smooth", "color", "gaussian"]
        # set to empty to not preprocess or add the methods in the order you want to implement.
        # "edge" to use edge detection, "gaussian" to use difference gaussian
        # "color" to use color segmentation, "smooth" to use smooth blurring,

    # color segmentation format
    HSV = True  # if true select a color which will convert to hsv format with a range of its own, else you can select your own rgb range
    f = lambda x: int(0) if (x < 0) else (int(255) if x > 255 else int(x))
    COLOR_HSV = [f(255), f(0), f(0)]     # target color which will be converted to hsv for processing, format BGR
    COLOR_LOW = (f(180), f(0), f(0))         # low color segment, format BGR
    COLOR_HIGH = (f(255), f(255), f(255))  # high color segment, format BGR

    # edge detection format
    INTENSITY_LOW = 50   # min 0, max 500
    INTENSITY_HIGH = 50  # min 0, max 500

    # smoothing_blurring
    GAUSSIAN_BLURRING = False
    KERNEL_SIZE = 15         # min 3, max 15
    STANDARD_DEVIATION = 0  # min 0.1, max 4.9

    # difference gaussian
    DIFFERENCE_SD_LOW = 1.5 # min 0.00, max 1.40
    DIFFERENCE_SD_HIGH = 0 # min 0.00, max 1.40
    ##########################
    """
    End of script settings
    """

    def reset_head_pose(self):
        """
        Reset MiRo head to default position, to avoid having to deal with tilted frames
        """
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(34.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():  # Check ROS is running
            # Publish state to neck servos for 1 sec
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break
        self.INTENSITY_CHECK = lambda x: int(0) if (x < 0) else (int(500) if x > 500 else int(x))
        self.KERNEL_SIZE_CHECK = lambda x: int(3) if (x < 3) else (int(15) if x > 15 else int(x))
        self.STANDARD_DEVIATION_PROCESS = lambda x: 0.1 if (x < 0.1) else (4.9 if x > 4.9 else round(x, 1))
        self.DIFFERENCE_CHECK = lambda x: 0.01 if (x < 0.01) else (1.40 if x > 1.40 else round(x,2))

    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)

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
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
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

    def detect_miro(self, frame_rgb, debug=False):
        if frame_rgb is None:
            return None

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Get largest white blob
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        radius = int(cv2.contourArea(largest) ** 0.5)

        if debug:
            cv2.drawContours(frame_rgb, [largest], -1, (0, 255, 0), 2)
            cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 255), -1)
            # cv2.imshow("White Body Detection", frame_rgb)
            cv2.waitKey(1)

        # Normalize
        h, w = frame_rgb.shape[:2]
        norm_x = (cx - w / 2) / w
        norm_y = (cy - h / 2) / w * -1.0
        norm_r = radius / w

        return [norm_x, norm_y, norm_r]
    
    def detect_blue_obstacle(self, frame_rgb, area_threshold=40000):
        if frame_rgb is None:
            return False
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = np.sum(mask > 0)
        return blue_pixels > area_threshold


    def detect_face_strict(self, image, debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        eye_candidates = []
        centers = []

        for c in contours:
            area = cv2.contourArea(c)
            if 30 < area < 300:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.6:  # 1.0 = perfect circle
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        eye_candidates.append(c)
                        centers.append((cx, cy))

        best_pair = None
        best_y_diff = float('inf')

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                y_diff = abs(cy1 - cy2)
                x_diff = abs(cx1 - cx2)
                if y_diff < 20 and 15 < x_diff < 120 and y_diff < best_y_diff:
                    best_pair = ((cx1, cy1), (cx2, cy2))
                    best_y_diff = y_diff

        if debug:
            debug_img = image.copy()
            cv2.drawContours(debug_img, eye_candidates, -1, (0, 255, 0), 2)
            for i, (cx, cy) in enumerate(centers):
                cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(debug_img, f"{i}", (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if best_pair:
                cv2.line(debug_img, best_pair[0], best_pair[1], (255, 0, 0), 2)

            cv2.imshow("Face Debug", debug_img)
            cv2.imshow("Threshold", thresh)
            cv2.waitKey(1)

        return best_pair is not None

    def look_for_miro(self):
        """
        [1 of 3] Roam forward, and bounce off obstacles by turning 180°.
        """
        if self.just_switched:
            print("MiRo is looking for the other MiRo...")
            self.just_switched = False

        # Check each camera for another MiRo
        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.target_miro[index] = self.detect_miro(image, index)

        if not self.target_miro[0] and not self.target_miro[1]:
            # No MiRo detected
            obstacle_detected = False
            if any(self.detect_blue_obstacle(img) for img in self.input_camera if img is not None):
                obstacle_detected = True
                rospy.loginfo("[VISION AVOID] Blue obstacle detected.")
            elif self.sonar_distance is not None and self.sonar_distance < self.SAFE_DISTANCE:
                obstacle_detected = True
                rospy.loginfo("[BOUNCE] Obstacle too close (sonar).")
            if obstacle_detected:
                # Turn in place: full spin, then move forward again
                start_time = rospy.Time.now().to_sec()
                while rospy.Time.now().to_sec() - start_time < self.TURN_DURATION / 2 and not rospy.core.is_shutdown():
                    self.drive(speed_l=self.TURNING_FACTOR, speed_r=-self.TURNING_FACTOR)
                    rospy.sleep(self.TICK)
                # After turning, move forward
                self.drive(speed_l=self.BASE_SPEED, speed_r=self.BASE_SPEED)
            else:
                # Nothing ahead, just move
                self.drive(speed_l=self.BASE_SPEED, speed_r=self.BASE_SPEED)
        else:
            # MiRo found!
            self.status_code = 2
            self.just_switched = True


    def lock_onto_miro(self):
        """
        Combined: Align with target and move forward only if aligned & distance is safe.
        """
        if self.just_switched:
            print("MiRo is aligning and following the other MiRo...")
            self.just_switched = False
            self.align_start_time = rospy.Time.now().to_sec()  # Start timing

        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.target_miro[index] = self.detect_miro(image, index)

        # Lost target
        if not self.target_miro[0] and not self.target_miro[1]:
            print("Target lost. Re-entering search mode...")
            self.status_code = 1
            self.just_switched = True
            return
        
        # Alignment timeout escape
        if rospy.Time.now().to_sec() - self.align_start_time > self.ALIGNMENT_TIMEOUT:
            print("[ALIGNMENT TIMEOUT] Giving up and switching to follow mode.")
            self.status_code = 3
            self.just_switched = True
            return

        # Alignment logic
        error_margin = 0.05
        rotation_speed = 0.03

        if self.target_miro[0] and self.target_miro[1]:
            left_x = self.target_miro[0][0]
            right_x = self.target_miro[1][0]
            diff = abs(left_x) - abs(right_x)

            if diff > error_margin:
                self.drive(rotation_speed, -rotation_speed)  # Clockwise
            elif diff < -error_margin:
                self.drive(-rotation_speed, rotation_speed)  # Counter-clockwise
            else:
                self.status_code = 3  # Switch to the second action
                self.just_switched = True
        
        # Only one MiRo detected by one camera
        elif self.target_miro[0]:
            self.drive(-rotation_speed, rotation_speed)
        elif self.target_miro[1]:
            self.drive(rotation_speed, -rotation_speed)

    def follow(self):
        """
        [3 of 3] MiRo moves forward for 1 second to simulate a 'follow',
        unless it's too close to an obstacle or a face is detected.
        If a face is detected, MiRo turns 180° instead of following.
        """

        if self.just_switched:
            print("MiRo is following the other miro!")
            self.just_switched = False
            self.follow_end_time = rospy.Time.now().to_sec() + 1.0  # 1 second from now
            return

        # Check for face detection
        # --- Face detection check with cooldown ---
        now = rospy.Time.now().to_sec()
        if now - self.last_face_detect_time > self.face_cooldown:
            for index in range(2):
                if self.new_frame[index]:
                    image = self.input_camera[index]
                    if self.detect_face_strict(image, debug=False):
                        rospy.loginfo("[FACE DETECTED] Turning 180° instead of following.")
                        self.last_face_detect_time = now  # prevent immediate retriggers
                        start_time = rospy.Time.now().to_sec()
                        while rospy.Time.now().to_sec() - start_time < self.TURN_DURATION and not rospy.core.is_shutdown():
                            self.drive(speed_l=self.TURNING_FACTOR, speed_r=-self.TURNING_FACTOR)
                            rospy.sleep(self.TICK)
                        self.status_code = 0
                        self.just_switched = True
                        self.follow_stop_counter = 0
                        return


        # Proceed with following if no face is detected
        if rospy.Time.now().to_sec() < self.follow_end_time:
            # --- Combined proximity check ---
            visual_close = False
            for i in range(2):
                if self.target_miro[i] and self.target_miro[i][2] > 0.1:  # radius threshold from detect_miro
                    visual_close = True
                    break

            sonar_close = self.sonar_distance is not None and self.sonar_distance < self.SAFE_DISTANCE

            if visual_close or sonar_close:
                # rospy.loginfo("[FOLLOW STOP] Too close! (Visual: %s, Sonar: %.3f m)", visual_close, self.sonar_distance or -1)
                rospy.loginfo("[FOLLOW STOP] Too close!")
                self.drive(0.0, 0.0)
                self.follow_stop_counter += 1
            else:
                self.drive(self.FAST, self.FAST)  # Move forward with a bit more speed
                self.follow_stop_counter = 0

            if self.follow_stop_counter >= self.follow_stop_limit:
                rospy.loginfo("[ESCAPE] Too many FOLLOW STOPs, turning 180° to reset.")
                start_time = rospy.Time.now().to_sec()
                while rospy.Time.now().to_sec() - start_time < self.TURN_DURATION and not rospy.core.is_shutdown():
                    self.drive(speed_l=self.TURNING_FACTOR, speed_r=-self.TURNING_FACTOR)
                    rospy.sleep(self.TICK)
                self.status_code = 0
                self.just_switched = True
                self.follow_stop_counter = 0
        else:
            self.status_code = 0
            self.just_switched = True


    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo, if needed
        if not self.IS_MIROCODE:
            rospy.init_node("boid_main", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)
        # Initialise CV Bridge
        self.image_converter = CvBridge()
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
        # Bookmark
        self.bookmark = 0
        # Move the head to default pose
        self.reset_head_pose()

        # Initialise the 'follow' escape threshold
        self.follow_stop_counter = 0
        self.follow_stop_limit = self.FOLLOW_STOP_LIMIT  # Frames before triggering escape

        self.last_face_detect_time = 0
        self.face_cooldown = self.FACE_DETECTION_COOLDOWN  # seconds to ignore face detection

    def sonar_callback(self, msg):
        self.sonar_distance = msg.range
        distance_cm = self.sonar_distance
        # rospy.loginfo_throttle(1.0, f"[Sonar] Distance: {distance_cm} cm")

    def loop(self):
        """
        Main control loop
        """
        print("MiRo boids-algorithm, press CTRL+C to halt...")
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find ball, lock on to the ball and follow ball
        self.status_code = 0
        while not rospy.core.is_shutdown():
            # Step 1. Find target MiRo
            if self.status_code == 1:
                # Every once in a while, look for ball
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_miro()

            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_miro()

            # Step 3. Follow!
            elif self.status_code == 3:
                self.follow()

            # Fall back
            else:
                self.status_code = 1

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)


# This condition fires when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop