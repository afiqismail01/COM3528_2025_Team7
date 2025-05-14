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

# NEW CODE 
from geometry_msgs.msg import Pose2D  # Used for broadcasting agent state
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TwistStamped
import csv
from datetime import datetime

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
            print("MiRo is looking for the other MiRo... " + str(self.robot_name))
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
                rospy.loginfo(f" [{self.robot_name}] [VISION AVOID] Blue obstacle detected.")
            elif self.sonar_distance is not None and self.sonar_distance < self.SAFE_DISTANCE:
                obstacle_detected = True
                rospy.loginfo(f" [{self.robot_name}][BOUNCE] Obstacle too close (sonar).")
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

        # NEW CODE 
        # Apply flocking even if no MiRo seen
        self.apply_boid_behavior()

    def lock_onto_miro(self):
        """
        Combined: Align with target and move forward only if aligned & distance is safe.
        """
        if self.just_switched:
            print("MiRo is aligning and following the other MiRo... " + str(self.robot_name))
            self.just_switched = False
            self.align_start_time = rospy.Time.now().to_sec()  # Start timing

        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.target_miro[index] = self.detect_miro(image, index)

        # Lost target
        if not self.target_miro[0] and not self.target_miro[1]:
            print("Target lost. Re-entering search mode... " + str(self.robot_name))
            self.status_code = 1
            self.just_switched = True
            return
        
        # Alignment timeout escape
        if rospy.Time.now().to_sec() - self.align_start_time > self.ALIGNMENT_TIMEOUT:
            print("[ALIGNMENT TIMEOUT] Giving up and switching to follow mode. " + str(self.robot_name))
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
            print("MiRo is following the other miro! " + str(self.robot_name))
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
                        rospy.loginfo(f" [{self.robot_name}] [FACE DETECTED] Turning 180° instead of following.")
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
                rospy.loginfo(f" [{self.robot_name}] [FOLLOW STOP] Too close!")
                self.drive(0.0, 0.0)
                self.follow_stop_counter += 1
            else:
                self.drive(self.FAST, self.FAST)  # Move forward with a bit more speed
                self.follow_stop_counter = 0

            if self.follow_stop_counter >= self.follow_stop_limit:
                rospy.loginfo(f" [{self.robot_name}] [ESCAPE] Too many FOLLOW STOPs, turning 180° to reset.")
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

    # NEW CODE 
    # def __init__(self):
    def __init__(self, miro_id='miro_1'):

        # NEW CODE
        self.miro_id = miro_id
        self.robot_name = os.getenv("MIRO_ROBOT_NAME", "miro01")
        print ("*************   __init__ function - " + str(self.robot_name))
        # topic_base = "/" + self.robot_name


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

        # NEW CODE
        # Boid-related
        self.pose = Pose2D()
        self.neighbors = []
        self.last_positions = {}
        self.boid_weight_sep = 1.5
        self.boid_weight_align = 1.0
        self.boid_weight_coh = 1.0
        self.boid_range = 1.0  # meters
        self.last_velocities = {}
        self.prev_position = None
        self.prev_time = None

        # ROS state pub/sub
        # self.state_pub = rospy.Publisher(
        #     f"{topic_base_name}/state", Pose2D, queue_size=1
        # )

        self.state_pub = rospy.Publisher(f"/{self.robot_name}/state", Pose2D, queue_size=1)
        for i in range(1, 6):  # assuming 5 agents: miro01 to miro05
            other = f"miro0{i}"
            if other != self.robot_name:
                rospy.Subscriber(f"/{other}/state", Pose2D, self.make_receive_neighbor_state(other))

        # rospy.Subscriber("/all_miro_states", Pose2D, self.receive_neighbor_state)

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)

        self.velocity_pub = rospy.Publisher(f"/{self.robot_name}/velocity", TwistStamped, queue_size=1)

        # for logging
        self.cohesion_log_file = f"/tmp/{self.robot_name}_cohesion_log.csv"
        self.log_file_handle = open(self.cohesion_log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file_handle)
        self.csv_writer.writerow(['timestamp', 'agent', 'num_neighbors', 'avg_pairwise_dist'])

        # END NEW CODE

    def sonar_callback(self, msg):
        self.sonar_distance = msg.range
        distance_cm = self.sonar_distance
        # rospy.loginfo_throttle(1.0, f"[Sonar] Distance: {distance_cm} cm")


    def model_state_callback(self, msg):
        matched = False
        for i, name in enumerate(msg.name):
            if self.robot_name in name:  # handles 'miro01' and 'miro01::base_link'
                position = msg.pose[i].position
                self.pose.x = position.x
                self.pose.y = position.y
                self.pose.theta = 0.0
                # print(f"&&&&&& [{self.robot_name}] matched {name} → pose: ({self.pose.x:.2f}, {self.pose.y:.2f})")
                # rospy.loginfo_throttle(2.0, f"[{self.robot_name}] matched {name} → x={position.x:.2f}, y={position.y:.2f}")
                matched = True
                break

        if not matched:
            rospy.logwarn_throttle(10.0, f"[{self.robot_name}] not found in /gazebo/model_states. Available: {msg.name}")


    # NEW CODE
    def make_receive_neighbor_state(self, sender_id):
        def callback(msg):
            self.last_positions[sender_id] = (msg.x, msg.y)
        return callback

    # NEW CODE
    def receive_neighbor_state(self, msg):
        robot_ns = rospy.get_namespace().strip("/")
        sender = msg._connection_header["callerid"].split("/")[1]
        if sender != robot_ns:
            self.last_positions[sender] = (msg.x, msg.y)
            # print(f"[{self.robot_name}] received pose from {sender}: ({msg.x:.2f}, {msg.y:.2f})")


    # NEW CODE 
    def receive_neighbor_velocity(self, msg):
        sender = msg._connection_header["callerid"].split("/")[1]
        if sender != self.robot_name:
            vx = msg.twist.linear.x
            vy = msg.twist.linear.y if hasattr(msg.twist, "linear") else 0.0
            self.last_velocities[sender] = np.array([vx, vy])


    # NEW CODE 
    def apply_boid_behavior(self):
        my_x, my_y = self.pose.x, self.pose.y
        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        count = 0

        for sender, (x, y) in self.last_positions.items():
            dx = x - my_x
            dy = y - my_y
            dist = np.hypot(dx, dy)
            if dist < 1e-5 or dist > self.boid_range:
                continue

            vec = np.array([dx, dy])
            cohesion += vec
            separation -= vec / (dist ** 2 + 1e-5)

            # Use velocity from the neighbor if available
            if sender in self.last_velocities:
                alignment += self.last_velocities[sender]
            else:
                alignment += vec / (dist + 1e-5)

            count += 1

        if count > 0:
            rospy.loginfo_throttle(2.0, f"[{self.robot_name}] Boid neighbors seen: {count}")
            cohesion = cohesion / count
            alignment = alignment / count

            steer = (self.boid_weight_coh * cohesion +
                    self.boid_weight_align * alignment +
                    self.boid_weight_sep * separation)

            norm = np.linalg.norm(steer)
            if norm > 1e-3:
                steer = steer / norm

                # Convert (vx, vy) → heading angle → turn control
                heading = np.arctan2(steer[1], steer[0])
                error = heading - self.pose.theta
                error = (error + np.pi) % (2 * np.pi) - np.pi  # Normalize

                # Smooth turn
                left = self.BASE_SPEED - 0.13 * error
                right = self.BASE_SPEED + 0.13 * error
                self.drive(left, right)

    # NEW CODE for logging

    def log_cohesion_metric(self):
        positions = list(self.last_positions.values())
        n = len(positions)
        if n < 2:
            print(f"[{self.robot_name}] Not enough neighbors to log cohesion (n={n})")
            return

        total_dist = 0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = (dx**2 + dy**2)**0.5
                total_dist += dist
                pairs += 1

        avg_dist = total_dist / pairs
        timestamp = rospy.Time.now().to_sec()
        self.csv_writer.writerow([timestamp, self.robot_name, n, avg_dist])
        self.log_file_handle.flush()

        rospy.loginfo_throttle(5.0, f"[{self.robot_name}] Cohesion avg distance: {avg_dist:.2f} (n={n})")

    def loop(self):
        """
        Main control loop
        """
        print("MiRo boids-algorithm, press CTRL+C to halt... " + str(self.robot_name))
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find ball, lock on to the ball and follow ball
        self.status_code = 0
        while not rospy.core.is_shutdown():

            # start New code 
            # Publish Gazebo position
            # print(f"[{self.robot_name}] Publishing pose: {self.pose.x:.2f}, {self.pose.y:.2f}")
            self.state_pub.publish(self.pose)

            vel = TwistStamped()
            now = rospy.Time.now().to_sec()
            if self.prev_position is not None and self.prev_time is not None:
                dt = now - self.prev_time
                if dt > 0:
                    vx = (self.pose.x - self.prev_position[0]) / dt
                    vy = (self.pose.y - self.prev_position[1]) / dt
                    vel.twist.linear.x = vx
                    vel.twist.linear.y = vy
            else:
                vel.twist.linear.x = 0.0
                vel.twist.linear.y = 0.0

            self.prev_position = (self.pose.x, self.pose.y)
            self.prev_time = now

            self.velocity_pub.publish(vel)
            # End New code 


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

            # new code for logging 
            self.log_cohesion_metric()



# This condition fires when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop