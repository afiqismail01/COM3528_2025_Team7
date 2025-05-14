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
    IS_MIROCODE = False  # Set to True if running in MiRoCODE
    
    # Fine-tune parameters below (depends on number of MiRo robots)
    ROTATION_SPEED = 0.1
    SAFE_DISTANCE = 0.13  # meters (adjust as needed)
    TURNING_FACTOR = 2  # Adjust this value to control the turning speed
    BASE_SPEED = 0.3  # Base speed for the robot
    FOLLOW_SPEED = 0.5
    TURN_DURATION = 1.8  # seconds for approx 180 turn
    SEPARATION_TIMEOUT = 1.0  # seconds to wait for separation before switching to alignment mode
    ALIGNMENT_TIMEOUT = 4.0  # seconds to wait for alignment before switching to follow mode
    FOLLOW_TIMEOUT = 0.5  # seconds to wait for follow before switching to alignment mode
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

        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        self.vel_pub.publish(msg_cmd_vel)

    def callback_caml(self, ros_image): 
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.input_camera[index] = image
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            self.new_frame[index] = True
        except CvBridgeError as e:
            pass

    def detect_miro(self, frame_rgb, index=0, debug=False):
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

        # Filter small contours
        MIN_AREA = 800 
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
                if circularity > 0.6:
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
            obstacle_detected = False
            if any(self.detect_blue_obstacle(img) for img in self.input_camera if img is not None):
                obstacle_detected = True
                # rospy.loginfo("[VISION AVOID] Blue obstacle detected.")
            elif self.sonar_distance is not None and self.sonar_distance < self.SAFE_DISTANCE:
                obstacle_detected = True
                # rospy.loginfo("[BOUNCE] Obstacle too close (sonar).")
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

    def face_detection_handler(self):
        """
        [3 of 3] Face detection check
        """
        if self.just_switched:
            self.just_switched = False
            rospy.loginfo(f"[{self.miro_name}] facing?")

        for index in range(2):
            if self.new_frame[index]:
                image = self.input_camera[index]
                if self.detect_face_strict(image, debug=False):
                    # rospy.loginfo("[FACE DETECTED] Turning 180° instead of following.")
                    start_time = rospy.Time.now().to_sec()
                    while rospy.Time.now().to_sec() - start_time < self.TURN_DURATION and not rospy.core.is_shutdown():
                        self.drive(speed_l=self.TURNING_FACTOR, speed_r=-self.TURNING_FACTOR)
                        rospy.sleep(self.TICK)
                    self.status_code = 0
                    self.just_switched = True
                    return       
        self.status_code = 3
        self.just_switched = True

    def steer_away(self):
        """
        [2 of 4] If the other Miro is too close, rotate MiRo away from it.
        """
        if self.just_switched:
            self.just_switched = False
            rospy.loginfo(f"[{self.miro_name}] separation")

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
        self.status_code = 4
        self.just_switched = True


    def lock_onto_miro(self, error=25):
        """
        [2 of 3] Once a ball has been detected, turn MiRo to face it
        """
        if self.just_switched:
            self.just_switched = False
            rospy.loginfo(f"[{self.miro_name}] alignment")

        for index in range(2): 
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.target_miro[index] = self.detect_miro(image, index)

        # Alignment whhen only one camera sees the MiRo
        if not self.target_miro[0] and self.target_miro[1]:
            self.drive(self.ROTATION_SPEED, -self.ROTATION_SPEED) # clockwise
        elif self.target_miro[0] and not self.target_miro[1]:
            self.drive(-self.ROTATION_SPEED, self.ROTATION_SPEED) # counter-clockwise

        # Detailed alignment when both cameras see the MiRo
        elif self.target_miro[0] and self.target_miro[1]:
            error = 0.25  # 25% of image width

            # Use the normalised values
            left_x = self.target_miro[0][0] 
            right_x = self.target_miro[1][0]
            rotation_speed = 0.03

            if abs(left_x) - abs(right_x) > error:
                self.drive(rotation_speed, -rotation_speed)  # turn clockwise
            elif abs(left_x) - abs(right_x) < -error:
                self.drive(-rotation_speed, rotation_speed)  # turn counter-clockwise
            else:
                # Successfully turned to face the ball
                self.status_code = 5  # Switch to the third action
                self.just_switched = True
                self.bookmark = self.counter
        # Otherwise, the ball is lost :-(
        else:
            self.status_code = 0  # Go back to square 1...
            # print("MiRo has lost the ball...")
            self.just_switched = True


    def follow(self):
        """
        [3 of 3] Once MiRO is in position, this function should drive the MiRo
        forward until it kicks the ball!
        """
        # --- Face detection check with cooldown ---
        for index in range(2):
            if self.new_frame[index]:
                image = self.input_camera[index]
                if self.detect_face_strict(image, debug=False):
                    # rospy.loginfo("[FACE DETECTED] Turning 180° instead of following.")
                    start_time = rospy.Time.now().to_sec()
                    while rospy.Time.now().to_sec() - start_time < self.TURN_DURATION and not rospy.core.is_shutdown():
                        self.drive(speed_l=self.TURNING_FACTOR, speed_r=-self.TURNING_FACTOR)
                        rospy.sleep(self.TICK)
                    self.status_code = 4
                    self.just_switched = True
                    return  
                    
        sonar_close = self.sonar_distance is not None and self.sonar_distance < self.SAFE_DISTANCE

        if self.just_switched:
            self.just_switched = False
            rospy.loginfo(f"[{self.miro_name}] follow")

        if not sonar_close:
            self.drive(self.FOLLOW_SPEED, self.FOLLOW_SPEED)
        else:
            self.status_code = 0  # Back to the default state after the kick
            self.just_switched = True


    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo, if needed
        if not self.IS_MIROCODE:
            rospy.init_node("boid_main", anonymous=True)

        rospy.sleep(2.0)
        self.image_converter = CvBridge()
        self.miro_name = os.getenv('MIRO_ROBOT_NAME')
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        self.vel_pub = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.sub_caml = rospy.Subscriber(topic_base_name + "/sensors/caml/compressed", CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(topic_base_name + "/sensors/camr/compressed", CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)
        sonar_topic = f"{topic_base_name}/sensors/sonar"
        rospy.Subscriber(sonar_topic, Range, self.sonar_callback)
        
        self.sonar_distance = None
        self.input_camera = [None, None]
        self.new_frame = [False, False]
        self.target_miro = [None, None]
        self.frame_width = 640
        self.just_switched = True
        self.bookmark = 0

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
        self.status_code = 0
        self.status_start_time = rospy.Time.now().to_sec()

        while not rospy.core.is_shutdown():
            now = rospy.Time.now().to_sec()
            # self.just_switched = True
            # self.status_code = 5
            # Step 1. Find target MiRo
            if self.status_code == 1:
                # Every once in a while, look for ball
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_miro()

            # Step 2. Align away from Miro
            elif self.status_code == 2:
                # now = rospy.Time.now().to_sec()

                # # 1. Wait until cooldown ends
                # if now - self.status_start_time < self.FACE_DETECT_COOLDOWN:
                #     # still in cooldown; do nothing
                #     pass
                # else:
                #     # After cooldown, begin face detection
                #     if now - self.status_start_time < self.FACE_DETECT_COOLDOWN + 1.0:
                #         self.status_code = 3
                #         self.status_start_time = now
                #         self.face_detection_handler()
                #     else:
                #         # If still in status 2 after 1s of trying, force status change
                #         self.status_code = 3
                #         self.status_start_time = now
                self.status_code = 3
                self.status_start_time = now

            # Step 3. Align away from Miro
            elif self.status_code == 3:
                if now - self.status_start_time > self.SEPARATION_TIMEOUT:
                    self.status_code = 4
                    self.status_start_time = now
                else:
                    self.steer_away()
                    if self.status_code == 4:
                        self.status_start_time = now

            # Step 4. Align towards Miro
            elif self.status_code == 4:
                if now - self.status_start_time > self.ALIGNMENT_TIMEOUT:
                    self.status_code = 5
                    self.status_start_time = now
                else:
                    self.lock_onto_miro()
                    if self.status_code == 5:
                        self.status_start_time = now

            # Step 5. Follow!
            elif self.status_code == 5:
                if now - self.status_start_time > self.FOLLOW_TIMEOUT:
                    self.status_code = 0
                    self.status_start_time = now
                else:
                    self.follow()
                    self.status_start_time = now

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