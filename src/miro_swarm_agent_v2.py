#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes MiRo look for a blue ball and kick it

The code was tested for Python 2 and 3
For Python 2 you might need to change the shebang line to
#!/usr/bin/env python
"""

# Imports
import os
import subprocess
from math import radians  # This is used to reset the head pose
import numpy as np  # Numerical Analysis library
import cv2  # Computer Vision library

import rospy  # ROS Python interface
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
# from geometry_msgs.msg import TwistStamped # ROS cmd_vel (velocity control) message

# new start
# from geometry_msgs.msg import TwistStamped, Point # ROS cmd_vel (velocity control) message
from geometry_msgs.msg import Twist, TwistStamped, Point, Pose2D
import miro2 as miro  # Import MiRo Developer Kit library
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2
import math

# new end
from std_msgs.msg import Float32MultiArray
import random

class MiRoSwarmAgent:

    """
    Script settings below
    """
    ##########################
    TICK = 0.02  # This is the update interval for the main control loop in secs
    CAM_FREQ = 1  # Number of ticks before camera gets a new frame, increase in case of network lag
    # SLOW = 0.1  # Radial speed when turning on the spot (rad/s)
    FAST = 0.4  # Linear speed when kicking the ball (m/s)
   

    # NEW SETTINGS *************
    # CAM_FREQ = 5  # Number of ticks before camera gets a new frame, increase in case of network lag
    SLOW = 0.4  # Radial speed when turning on the spot (rad/s)
    # FAST = 0.8  # Linear speed when kicking the ball (m/s)
   


    DEBUG = False # Set to True to enable debug views of the cameras
    TRANSLATION_ONLY = False # Whether to rotate only
    IS_MIROCODE = False  # Set to True if running in MiRoCODE

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

    # new for Obstacle
    AVOID_ENTRY = 0.15   # when |offset| drops below this → enter avoid
    AVOID_EXIT  = 0.30   # when |offset| rises above this → exit avoid

    
    # OBS_MIN_AREA     = 2000    # min pixel area of a real ball blob
    OBS_MIN_AREA     = 5000    # min pixel area of a real ball blob

    OBS_MAX_AREA     = 30000   # max pixel area of a real ball blob
    OBS_CIRC_THRESH  = 0.65    # only accept contours with circularity > 0.65

    MIN_DETECT_R    = 0.02   # normalized radius threshold
    SEARCH_ROT_TICKS= 50     # how many loop ticks to spin before driving
    SEARCH_FWD_TICKS= 200    # how many ticks to drive straight

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

        # new
        self.velocity = msg_cmd_vel.twist

        # old code commented
        # Publish message to control/cmd_vel topic
        # self.vel_pub.publish(msg_cmd_vel)

        # … compute msg_cmd_vel …
        self.cmd_pub.publish(msg_cmd_vel)

    # NEW START SWARM

    def odom_callback(self, msg):
        """Keep self.position up to date from real odometry."""
        self.position = msg.pose.pose.position

    def pose2d_callback(self, msg: Pose2D):
        """Update self.position from Gazebo’s Pose2D message."""
        self.position.x = msg.x
        self.position.y = msg.y
        self.yaw        = msg.theta
        # z stays at whatever you set (0.05), or reset if you like:
        # self.position.z = 0.05

    # def _model_states_cb(self, msg: ModelStates):
    #     try:
    #         i = msg.name.index("ball_1")
    #         self.ball_position = msg.pose[i].position
    #     except ValueError:
    #         pass
        
    def _leader_pose_callback(self, msg: Pose2D):
        # keep track of where the leader is pointing
        self.leader_yaw = msg.theta


    def prox_callback(self, msg):
        self.prox = msg.data

    def make_pos_cb(self, name):
        def cb(msg):
            self.neighbors[name]['pos'] = msg

            # rospy.loginfo(f"[{self.robot_name}] got {name} pos → {msg.x:.3f}, {msg.y:.3f}")
    
        return cb

    def make_vel_cb(self, name):
        def cb(msg):
            self.neighbors[name]['vel'] = msg
        return cb   
    
    def compute_separation(self):
        force = Point()

        for n in self.neighbors.values():
            dx = n['pos'].x - self.position.x
            dy = n['pos'].y - self.position.y
            dist_sq = dx*dx + dy*dy
            # repel from anyone within SEP_MAX_DIST (e.g. 1.5 m)
            SEP_MAX_DIST = 1.5
            if dist_sq < SEP_MAX_DIST*SEP_MAX_DIST and dist_sq > 1e-6:
                # stronger repulsion when very close
                force.x -= dx / dist_sq
                force.y -= dy / dist_sq
        return force


    def compute_alignment(self):
        avg_vel = Point()
        count = 0
        for name, data in self.neighbors.items():
            w = 2.0 if name == "miro01" else 1.0  # weight leader more
            avg_vel.x += w * data['vel'].twist.linear.x
            avg_vel.y += w * data['vel'].twist.linear.y
            count += w
        if count > 0:
            avg_vel.x /= count
            avg_vel.y /= count
        return avg_vel


    def compute_cohesion(self):
        center = Point()
        count = 0
        for name, data in self.neighbors.items():
            w = 2.0 if name == "miro01" else 1.0
            center.x += w * data['pos'].x
            center.y += w * data['pos'].y
            count += w
        if count > 0:
            center.x /= count
            center.y /= count
        force = Point()
        force.x = center.x - self.position.x
        force.y = center.y - self.position.y
        return force


    # NEW END 

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

    def detect_ball(self, frame, index):
        """
        Image processing operations, fine-tuned to detect a small,
        toy blue ball in a given frame.
        """
        if frame is None:  # Sanity check
            return

        # Debug window to show the frame
        if self.DEBUG:
            cv2.imshow("camera" + str(index), frame)
            cv2.waitKey(1)

        # Flag this frame as processed, so that it's not reused in case of lag
        self.new_frame[index] = False

        processed_img = frame

        for method in self.PREPROCESSING_ORDER:
            if method == "color":
                if self.HSV == True:
                    # Get image in HSV (hue, saturation, value) colour format
                    im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

                    # Specify target ball colour
                    rgb_colour = np.uint8([[self.COLOR_HSV]])  # e.g. Blue (Note: BGR)
                    # Convert this colour to HSV colour model
                    hsv_colour = cv2.cvtColor(rgb_colour, cv2.COLOR_RGB2HSV)

                    # Extract colour boundaries for masking image
                    # Get the hue value from the numpy array containing target colour
                    target_hue = hsv_colour[0, 0][0]
                    hsv_lo_end = np.array([target_hue - 20, 70, 70])
                    hsv_hi_end = np.array([target_hue + 20, 255, 255])

                    # Generate the mask based on the desired hue range
                    mask = cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)
                    processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)
                else:
                    mask = cv2.inRange(frame, self.COLOR_LOW, self.COLOR_HIGH)
                    processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)

            elif method == "gaussian":
                sigma1 = self.DIFFERENCE_CHECK(self.DIFFERENCE_SD_LOW)
                sigma2 = self.DIFFERENCE_CHECK(self.DIFFERENCE_SD_HIGH)
                img_gauss1 = cv2.GaussianBlur(processed_img, (0, 0), sigma1)
                img_gauss2 = cv2.GaussianBlur(processed_img, (0, 0), sigma2)
                processed_img = img_gauss1 - img_gauss2

            elif method == "smooth":
                kernel_size = (self.KERNEL_SIZE_CHECK(self.KERNEL_SIZE), self.KERNEL_SIZE_CHECK(self.KERNEL_SIZE))

                if not self.GAUSSIAN_BLURRING:
                    # average smoothing
                    kernel = np.ones(kernel_size, np.float32) / kernel_size[0]**2
                    processed_img = cv2.filter2D(processed_img, -1, kernel)

                else:
                    # Gaussian blurring
                    sigma = self.STANDARD_DEVIATION_PROCESS(self.STANDARD_DEVIATION)
                    processed_img = cv2.GaussianBlur(processed_img, (0,0), sigma) # kernel size computed as: [(sigma - 0.8)/0.3 + 1] / 0.5 + 1
                                                                    # see opencv documentation

            elif method == "edge":
                processed_img = cv2.Canny(processed_img, self.INTENSITY_LOW, self.INTENSITY_HIGH)

        # Debug window to show the mask
        if self.DEBUG:
            cv2.imshow("mask" + str(index), processed_img)
            cv2.waitKey(1)

        if len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        # Debug window to show the mask
        if self.DEBUG:
            cv2.imshow("gray" + str(index), processed_img)
            cv2.waitKey(1)

        # Fine-tune parameters
        ball_detect_min_dist_between_cens = 40  # Empirical
        canny_high_thresh = 10  # Empirical
        ball_detect_sensitivity = 10  # Lower detects more circles, so it's a trade-off

        # check
        # ball_detect_min_radius = 5  # Arbitrary, empirical
        # ball_detect_max_radius = 50  # Arbitrary, empirical

        # NEW SETTINGS
        ball_detect_min_radius = 2  # Arbitrary, empirical
        ball_detect_max_radius = 100 # Arbitrary, empirical  

        # Find circles using OpenCV routine
        # This function returns a list of circles, with their x, y and r values
        circles = cv2.HoughCircles(
            processed_img,
            cv2.HOUGH_GRADIENT,
            1,
            ball_detect_min_dist_between_cens,
            param1=canny_high_thresh,
            param2=ball_detect_sensitivity,
            minRadius=ball_detect_min_radius,
            maxRadius=ball_detect_max_radius,
        )

        if circles is None:
            # If no circles were found, just display the original image
            return

        # Get the largest circle
        max_circle = None
        self.max_rad = 0
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            if c[2] > self.max_rad:
                self.max_rad = c[2]
                max_circle = c
        # This shouldn't happen, but you never know...
        if max_circle is None:
            return

        # Append detected circle and its centre to the frame
        cv2.circle(frame, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 0), 2)
        cv2.circle(frame, (max_circle[0], max_circle[1]), 2, (0, 0, 255), 3)
        if self.DEBUG:
            cv2.imshow("circles" + str(index), frame)
            cv2.waitKey(1)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]
        max_circle = np.array(max_circle).astype("float32")
        max_circle[0] -= self.x_centre
        max_circle[0] /= self.frame_width
        max_circle[1] -= self.y_centre
        max_circle[1] /= self.frame_width
        max_circle[1] *= -1.0
        max_circle[2] /= self.frame_width


        # START NEW CODE FOR OBSTACLE
        # --- Visual Obstacle Detection (non-blue mask) ---
        self.process_red_obstacle_from_image(frame)
        # END

        # Return a list values [x, y, r] for the largest circle
        return [max_circle[0], max_circle[1], max_circle[2]]


    #  NEW instead of non blue.. specifically look for RED
    def process_red_obstacle_from_image(self, frame):
        """
        Detect a red obstacle in the bottom part of the frame,
        compute its normalized x-offset in [-1,1], and store in
        self.obstacle_offset. If none found, offset = 0.
        """
        self.obstacle_offset = 0.0
        if frame is None:
            return

        # Convert RGB frame to HSV
        im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        if self.DEBUG:
            cv2.imshow("red", frame)
            cv2.waitKey(1)

        h, w = im_hsv.shape[:2]
        # CHECK THIS
        # roi = im_hsv[int(h*0.4):, :]  # bottom 60%
        roi = im_hsv[int(h*0.8):, :]  # bottom 20%

        # 1) threshold for red (two ranges)
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(roi, lower1, upper1)
        mask2 = cv2.inRange(roi, lower2, upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 2) clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # 3) find contours, pick the largest by area
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.OBS_MIN_AREA or area > self.OBS_MAX_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            circ = 4 * math.pi * area / (peri*peri)
            if circ < self.OBS_CIRC_THRESH:
                continue
            x,y,w_rect,h_rect = cv2.boundingRect(cnt)
            ar = w_rect/float(h_rect) if h_rect>0 else 0
            if not (0.8 <= ar <= 1.2):
                continue
            if circ > best_score:
                best_score = circ
                best = cnt

        if best is not None:
            M = cv2.moments(best)
            cx = int(M['m10']/M['m00'])
            # now compute offset in [-1,1]
            self.obstacle_offset = (cx - w/2) / (w/2)
            if self.just_switched:  # Print once
                self.just_switched = False
                # rospy.loginfo(f"[{self.robot_name}] RED obstacle offset: {self.obstacle_offset:.2f}")
   
        else:
            self.obstacle_offset = 0.0

    # OLD appraoch to start look for NON-blue obstacle
    # this didn't work well as it was identifying floor and other things as obstacle
    def process_obstacle_from_image(self, frame):
        try:

            if frame is None:
                self.obstacle_offset = 0.0
                return
            
            # im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            height, width = im_hsv.shape[0:2]
            # look at bottom 60%
            roi_start = int(height * 0.4)
            roi = im_hsv[roi_start:, :]
            # roi = im_hsv  # no cropping

            # Blue mask
            rgb_colour = np.uint8([[self.COLOR_HSV]])
            hsv_colour = cv2.cvtColor(rgb_colour, cv2.COLOR_RGB2HSV)
            target_hue = hsv_colour[0, 0][0]
            lower_blue = np.array([max(0, target_hue - 20), 70, 70])
            upper_blue = np.array([min(179, target_hue + 20), 255, 255])
            blue_mask = cv2.inRange(roi, lower_blue, upper_blue)

            # below works fine in Football arena.. world and not in groung_plane arena in world
            non_blue_mask = cv2.bitwise_not(blue_mask)
            contours, _ = cv2.findContours(non_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # new approach
            # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # # Detect dark regions (e.g., cubes)
            # _, obstacle_mask = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY_INV)  # Tune threshold
            # contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # approach2 - failed Directly detect dark/grey objects in HSV (instead of inverting blue)
            # lower_obstacle = np.array([0, 0, 30])    # Adjust as needed
            # upper_obstacle = np.array([180, 50, 100])  # Low saturation = greys
            # obstacle_mask = cv2.inRange(im_hsv, lower_obstacle, upper_obstacle)
            # contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            self.obstacle_offset = 0.0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 500  < area < 50000:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        offset = (cx - width / 2) / (width / 2)
                        self.obstacle_offset = offset
                        rospy.loginfo(f"[{self.robot_name}] Obstacle offset updated: {offset:.2f}")
                        break
        except Exception as e:
            rospy.logwarn(f"[{self.robot_name}] Obstacle detection error: {e}")
            self.obstacle_offset = 0.0


    # END

    def look_for_ball(self):
        """
        [1 of 3] Rotate MiRo if it doesn't see a ball in its current
        position, until it sees one.
        """

        # clear any previous detections
        # self.ball = [None, None]


        if self.just_switched:  # Print once
            print("MiRo " + str(self.robot_name) + " started looking for the ball...")
            self.just_switched = False
        
       
        # init on first entry
        # if self.look_for_ball_again:
        #     # print(f"MiRo {self.robot_name} look_for_ball_again- entering search phase")
        #     self.search_phase = 'rotate'
        #     self.search_timer = 0
        #     # flip spin direction every search
        #     self.search_dir *= -1
        #     self.just_switched = False
        #     self.look_for_ball_again = False

        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect ball procedure
            self.ball[index] = self.detect_ball(image, index)

        if not self.ball[0] and not self.ball[1]:

            # self.drive(self.SLOW, -self.SLOW)

            self.search_timer += 1
            if self.search_timer >= self.SEARCH_ROT_TICKS:
                # self.drive(self.SLOW, -self.SLOW)
                # Spiral search: forward + rotation
                search_fwd = 0.1              # linear m/s
                search_rot = 0.3              # rad/s
                left_wheel  = search_fwd + search_rot
                right_wheel = search_fwd - search_rot
                # clamp so we don’t exceed FAST
                left_wheel  = max(min(left_wheel,  self.FAST), -self.FAST)
                right_wheel = max(min(right_wheel, self.FAST), -self.FAST)
                self.drive(left_wheel, right_wheel)
            else:
                self.search_timer  = 0
                fwd_speed = min(self.FAST, 0.4)
                self.drive(fwd_speed, fwd_speed)

            # if self.just_switched:  # Print once
            #     print("MiRo " + str(self.robot_name) + " still looking for the ball...")
            #     self.just_switched = False

            # print("self.search_phase" + str(self.search_phase))

            # 4) two‐phase: rotate then forward
            # if self.search_phase == 'rotate':
            #     # spin in place
            #     spin = self.SLOW * self.search_dir
            #     self.drive(-spin, +spin)
            # self.search_timer += 1
            # if self.search_timer >= self.SEARCH_ROT_TICKS:
            #     self.search_phase  = 'forward'
            #     self.search_timer  = 0
            #     fwd_speed = min(self.FAST, 0.4)
            #     self.drive(fwd_speed, fwd_speed)
            #     self.search_timer += 1
            #     if self.search_timer >= self.SEARCH_FWD_TICKS:
            #         self.search_phase  = 'rotate'
            #         self.search_timer  = 0

        else:
            # locking mode
            self.status_code = 2  # Switch to the second action
            self.just_switched = True



    def lock_onto_ball(self, error=25):
        """
        [2 of 3] Once a ball has been detected, turn MiRo to face it
        """
        if self.just_switched:  # Print once
            print("MiRo " + str(self.robot_name) + " is locking on to the ball...")
            self.just_switched = False

        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect ball procedure
            self.ball[index] = self.detect_ball(image, index)

        # If only the right camera sees the ball, rotate clockwise
        if not self.ball[0] and self.ball[1]:
            self.drive(self.SLOW, -self.SLOW)
        # Conversely, rotate counter-clockwise
        elif self.ball[0] and not self.ball[1]:
            self.drive(-self.SLOW, self.SLOW)
        # Make the MiRo face the ball if it's visible with both cameras
        elif self.ball[0] and self.ball[1]:
            error = 0.05  # 5% of image width
            # Use the normalised values
            left_x = self.ball[0][0]  # should be in range [0.0, 0.5]
            right_x = self.ball[1][0]  # should be in range [-0.5, 0.0]
            rotation_speed = 0.03  # Turn even slower now
            if abs(left_x) - abs(right_x) > error:
                self.drive(rotation_speed, -rotation_speed)  # turn clockwise
            elif abs(left_x) - abs(right_x) < -error:
                self.drive(-rotation_speed, rotation_speed)  # turn counter-clockwise
            else:
                # Successfully turned to face the ball
                # travel mode
                self.status_code = 3  # Switch to the third action
                self.just_switched = True
                self.bookmark = self.counter
                self.look_for_ball_again = False
        # Otherwise, the ball is lost :-(
        else:
            self.status_code = 0  # Go back to square 1...
            print("MiRo " + str(self.robot_name) + "has lost the ball...")
            self.look_for_ball_again = True
            self.just_switched = True

    # GOAAAL
    def kick(self):
        """
        [3 of 3] Once MiRO is in position, this function should drive the MiRo
        forward until it kicks the ball!
        """
        if self.just_switched:
            print("MiRo " + str(self.robot_name) + "is kicking the ball!")
            self.just_switched = False
        if self.counter <= self.bookmark + 2 / self.TICK and not self.TRANSLATION_ONLY:
            self.drive(self.FAST, self.FAST)
        else:
            self.status_code = 0  # Back to the default state after the kick
            self.just_switched = True

    def traveltodestination_emergent_flocking(self):
        """
        Emergent flocking for leader + followers.
        Leader has a vision-based goal term; followers use boid alignment+cohesion.
        Obstacle bump and separation always apply.
        """

        # if self.robot_name == 'miro01':
        #     if self.counter <= self.bookmark + 2 / self.TICK and not self.TRANSLATION_ONLY:
        #         self.drive(self.FAST, self.FAST)
        #     else:
        #         self.status_code = 0  # Back to the default state after the kick
        #         self.just_switched = True
        #     return

        if self.robot_name=='miro01':
            if self.just_switched:
                print("MiRo " + str(self.robot_name) + "moving towards the blue ball!")
                self.just_switched = False


            # TEST WHICH OPTION IS BETTER

            # option 1
            # for index in range(2):  # For each camera (0 = left, 1 = right)
            #     # Skip if there's no new image, in case the network is choking
            #     if not self.new_frame[index]:
            #         continue
            #     image = self.input_camera[index]
            #     # Run the detect ball procedure
            #     self.ball[index] = self.detect_ball(image, index)

            # option 2
            # for i in (0,1):
            #     img = self.input_camera[i]
            #     if img is not None:
            #         self.ball[i] = self.detect_ball(img, i)
            #     else:
            #         self.ball[i] = None


            # option 3
            # —— 1) re-detect ball only on fresh frames, and only every DETECT_INTERVAL ticks
            DETECT_INTERVAL = 3
            if (self.counter % DETECT_INTERVAL)==0:
                for i in (0,1):
                    if self.new_frame[i]:
                        img = self.input_camera[i]
                        self.ball[i] = self.detect_ball(img, i) if img is not None else None


            # —— 2) if leader lost the ball, go back to search
            seen = [b for b in self.ball if b and b[2] > self.MIN_DETECT_R]
            # if len(seen) < 1:
            if not seen:
                print("traveltodestination_emergent_flocking - MiRo - " + str(self.robot_name) + " Detected FAILED...")
                # not seeing it clearly → switch back to look/search
                self.status_code = 1
                self.just_switched = True
                return
            else:
                print("traveltodestination_emergent_flocking - MiRo - " + str(self.robot_name) + " Detected again...")
                
           
        if self.robot_name=='miro01':
            print("traveltodestination_emergent_flocking - MiRo - " + str(self.robot_name) + " MOVING...")

        # —— 3) obstacle detection
        self.process_red_obstacle_from_image(self.input_camera[0])

        if self.robot_name=='miro01':
            if abs(self.obstacle_offset)<self.AVOID_ENTRY and self.obstacle_offset!=0:
            # if abs(self.obstacle_offset)<self.AVOID_ENTRY:
                self.avoid_dir = -1 if self.obstacle_offset>0 else 1
                self.status_code=4
                self.just_switched=True
                return

        # —— 4) compute separation (always)
        sep = self.compute_separation()

        # —— 5) leader goal-attraction term
        goal_vx_r = 0.0
        goal_vy_r = 0.0
        if self.robot_name=='miro01':
            # constant forward
            goal_vx_r = 0.2
            # steer toward average x-offset of any seen ball
            offsets = [b[0] for b in self.ball if b]
            if offsets:
                bx = sum(offsets)/len(offsets)
                goal_vy_r = -0.5 * bx

        # —— 6) boid alignment+cohesion for followers only
        if self.robot_name!='miro01':
            ali = self.compute_alignment()
            coh = self.compute_cohesion()
        else:
            ali = Point(0,0,0)
            coh = Point(0,0,0)

        # —— 7) rotate robot-frame goal into world frame
        c,s = math.cos(self.yaw), math.sin(self.yaw)
        goal_vx_w =  c*goal_vx_r - s*goal_vy_r
        goal_vy_w =  s*goal_vx_r + c*goal_vy_r

        # —— 8) sum components + obstacle bump
        vx = 1.5*sep.x + (1.0*ali.x if self.robot_name!='miro01' else 0) + \
             (1.0*coh.x if self.robot_name!='miro01' else 0) + goal_vx_w
        vy = 1.5*sep.y + (1.0*ali.y if self.robot_name!='miro01' else 0) + \
             (1.0*coh.y if self.robot_name!='miro01' else 0) + goal_vy_w

        # obstacle pushes
        # increase for sharper turns, or down for gentler sweep.
        vx += -3.0 * self.obstacle_offset

        # —— 9) differential-drive conversion
        speed   = min(math.hypot(vx,vy),  self.FAST*0.75)
        desired = math.atan2(vy, vx)
        err     = ((desired - self.yaw + math.pi) % (2*math.pi)) - math.pi
        ω       = err
        tw      = 0.13
        left_wheel  = speed - (tw/2)*ω
        right_wheel = speed + (tw/2)*ω

        # clamp to ±FAST
        left_wheel  = max(min(left_wheel,  self.FAST), -self.FAST)
        right_wheel = max(min(right_wheel, self.FAST), -self.FAST)

        # —— 10) send commands & publish
        self.drive(left_wheel, right_wheel)
        self._publish_swarm_state()

     
    # def __init__(self):
    def __init__(self, agent_id='agent_1'):

        # Initialise a new ROS node to communicate with MiRo, if needed
        if not self.IS_MIROCODE:
            rospy.init_node("swarm_agent_node", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)



        #  START new code for swarm
             
        self.agent_id = agent_id
        self.robot_name = os.getenv("MIRO_ROBOT_NAME", "miro01")
        topic_base = "/" + self.robot_name

        # self.position = Point(random.uniform(0, 2), random.uniform(0, 2), 0)
        # Initialise to a known pose until the first body_pose callback arrives
        self.position = Point(0.0, 0.0, 0.05)

        self.yaw = 0.0

        if self.robot_name != 'miro01':
            self.leader_yaw = 0.0 

        # Publishers
        self.cmd_pub = rospy.Publisher(topic_base + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.pub_kin = rospy.Publisher(topic_base + "/control/kinematic_joints",JointState,     queue_size=0)
       
        self.pos_pub = rospy.Publisher(f"{topic_base}/agent_position", Point, queue_size=10)
        self.state_vel_pub = rospy.Publisher(topic_base + "/agent_velocity", TwistStamped,   queue_size=10)

        # self.vel_pub = rospy.Publisher(f"{topic_base}/agent_velocity", TwistStamped, queue_size=10)

        # self.odom_sub = rospy.Subscriber(topic_base + "/sensors/odom", Odometry, self.odom_callback)
        self.pose2d_sub  = rospy.Subscriber(topic_base + "/sensors/body_pose",Pose2D, self.pose2d_callback)
       
        #    subscribe to the leader’s body_pose 
        if self.robot_name != 'miro01':
            self.leader_yaw = 0.0
            rospy.Subscriber("/miro01/sensors/body_pose", Pose2D, self._leader_pose_callback)



        # Subscribers
        self.prox = [0.0, 0.0]
        rospy.Subscriber(topic_base + "/sensors/proximity", Float32MultiArray, self.prox_callback)
        # rospy.Subscriber(topic_base + "/sensors/caml/compressed", CompressedImage, self.cam_callback)

        # State exchange with neighbors
        self.neighbors = {}
        for name in ["miro01", "miro02", "miro03"]:
            if name != self.robot_name:
                self.neighbors[name] = {'pos': Point(), 'vel': TwistStamped()}
                rospy.Subscriber(f"/{name}/agent_position", Point, self.make_pos_cb(name))
                rospy.Subscriber(f"/{name}/agent_velocity", TwistStamped, self.make_vel_cb(name))

        # Internal state
        # self.image_converter = CvBridge()
        self.latest_image = None


        self.velocity = Twist()
        self.blue_seen = False
        self.red_seen = False
        # self.rate = rospy.Rate(10)
        self.last_radii = []
        self.obstacle_offset = 0.0



        # self.blue_offset = 0
        self.avoid_timer = 0
        self.leader_looking = True

        # self.celebrate_timer = 0

        # self.ball_position = Point()
        # rospy.Subscriber("/gazebo/model_states",
        #                  ModelStates,
        #                  self._model_states_cb,
        #                  queue_size=1)

        # for robust arrival detection
        self.arrival_counter   = 0
        self.arrival_required  = 12   # e.g. 5 consecutive ticks
        # self.arrival_radius_thr= 0.14 # tune this to just below your plateau        
        self.arrival_radius_thr= 0.05 # tune this to just below your plateau        

        self.search_phase  = 'rotate'
        self.search_timer  = 0
        self.search_dir    = 1  # will alternate spin direction
        self.look_for_ball_again = True

        # for debouncing “lost ball”
        self.lost_counter    = 0

        #  END

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
        # new original code commented
        # # Create a new publisher to send velocity commands to the robot
        # self.vel_pub = rospy.Publisher(
        #     topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        # )
        # # Create a new publisher to move the robot head
        # self.pub_kin = rospy.Publisher(
        #     topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        # )
        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of ball's x, y, and r values for each camera
        self.ball = [None, None]
        # Set the default frame width (gets updated on receiving an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True
        # Bookmark
        self.bookmark = 0
        # Move the head to default pose
        self.reset_head_pose()

    def _publish_swarm_state(self):
        """Publish pos & vel to neighbors."""
        self.pos_pub.publish(self.position)
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist = self.velocity
        self.state_vel_pub.publish(msg)

    def loop(self):
        """
        Main control loop
        """
        # print("MiRo finds ball, press CTRL+C to halt...")
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find ball, lock on to the ball and kick ball
        self.status_code = 0

        self.logcounter = 0

        # Leader searches for ball; followers go straight to travel
        if self.robot_name == 'miro01':
            # look for mode
            self.status_code = 1
        else:
            # travel mode
            self.status_code = 3

        while not rospy.core.is_shutdown():

            # NEW

            if self.robot_name != 'miro01':
                # self.traveltodestination()
                self.traveltodestination_emergent_flocking()
                # publish state and continue
                self._publish_swarm_state()
                self.counter += 1
                rospy.sleep(self.TICK)
                continue

            # end new

            if self.status_code == -1:
                 self.drive(0.0, 0.0)
                 return

            # Step 1. Find ball
            if self.status_code == 1:
                # Every once in a while, look for ball
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_ball()

            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_ball()

            # Step 3. traveltodestination!
            elif self.status_code == 3:
                # self.kick()
                # if self.just_switched:
                #     print("MiRo - " + str(self.robot_name) + " moving towards the Blue ball...")
                #     self.just_switched = False
                # self.traveltodestination()
                self.traveltodestination_emergent_flocking()

            # Step 4. Obstacle avoidance mode
            elif self.status_code == 4:
                if self.just_switched:
                    rospy.loginfo(f"[{self.robot_name}] Entering avoidance mode offset={self.obstacle_offset:.2f}")
                    self.avoid_timer = 0
                    self.just_switched = False

                forward_speed = 0.05  # small forward push
                turn_speed    = self.SLOW

                # print("self.avoid_dir " + str(self.avoid_dir))
                # print("self.obstacle_offset " + str(self.obstacle_offset))

                if self.avoid_dir < 0:
                    # obstacle on right → arc left
                    left_wheel  = forward_speed - turn_speed
                    right_wheel = forward_speed + turn_speed
                else:
                    # obstacle on left → arc right
                    left_wheel  = forward_speed + turn_speed
                    right_wheel = forward_speed - turn_speed

                 # clamp to ±FAST
                left_wheel  = max(min(left_wheel,  self.FAST), -self.FAST)
                right_wheel = max(min(right_wheel, self.FAST), -self.FAST)

                self.drive(left_wheel, right_wheel)

                # exit once the blob is well out of center
                if abs(self.obstacle_offset) > self.AVOID_EXIT:
                    # rospy.loginfo(f"[{self.robot_name}] Cleared obstacle offset={self.obstacle_offset:.2f} — resuming")
                    rospy.loginfo(f"[{self.robot_name}] Cleared obstacle offset={self.obstacle_offset:.2f}")
  
                    self.status_code   = 3    # back to “go to goal”
                    self.just_switched = True

                continue          

            # NCB need to handle this case where we set self.status_code = 10
            # in traveltodestination when Miro reaches the blue ball
            elif self.status_code == 10:  # celebration
                if self.celebrate_timer < 10:
                    if self.celebrate_timer % 2 == 0:
                        self.drive(0.2, -0.2)  # left wiggle
                    else:
                        self.drive(-0.2, 0.2)  # right wiggle
                    self.celebrate_timer += 1
                else:
                    self.drive(0.0, 0.0)
                    print(f"[{self.robot_name}] Reached GOAL!!!!!!")
                    self.status_code = -1  # done
            # Fall back
            else:
                self.status_code = 1

            # Enable logging after certain count
            self.logcounter +=1
            if self.logcounter > 100:
                self.just_switched = True  
                self.logcounter = 0              


            self._publish_swarm_state() 
            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)


# This condition fires when the script is called directly
if __name__ == "__main__":
    # main = MiRoSwarmAgent()  # Instantiate class
    # main.loop()  # Run the main control loop
    try:
        agent = MiRoSwarmAgent()
        agent.loop()
    except rospy.ROSInterruptException:
        pass