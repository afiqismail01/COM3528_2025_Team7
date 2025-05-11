#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

try:
    from miro2.lib import wheel_speed2cmd_vel
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel

class MiroClient:
    """
    Detects MiRo face (eyes) using camera input.
    """

    ##########################
    TICK = 0.02
    IS_MIROCODE = False  # Set to True if running in MiRoCODE
    ##########################

    def __init__(self):
        if not self.IS_MIROCODE:
            rospy.init_node("face_detector", anonymous=True)

        # Init CvBridge
        self.bridge = CvBridge()

        # Subscribe to left camera image
        miro_name = os.getenv("MIRO_ROBOT_NAME")
        cam_topic = f"/{miro_name}/sensors/caml/compressed"
        self.sub_cam = rospy.Subscriber(cam_topic, CompressedImage, self.callback_cam, queue_size=1)

        self.latest_image = None

        rospy.sleep(2.0)

    def callback_cam(self, ros_image):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image, desired_encoding="rgb8")
            self.latest_image = image
        except Exception as e:
            rospy.logwarn(f"[Camera Error] {e}")
            self.latest_image = None

    def detect_miro(self, frame_rgb, debug=False):
        if frame_rgb is None:
            return False

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return False

        if debug:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawContours(frame_rgb, [largest], -1, (0, 255, 0), 2)
            cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 255), -1)
            cv2.imshow("White Body Detection", frame_rgb)
            cv2.waitKey(1)

        return True


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





    def loop(self):
        rospy.loginfo("MiRo face detection active...")
        rate = rospy.Rate(1 / self.TICK)

        while not rospy.core.is_shutdown():
            if self.latest_image is not None:
                face_detected = self.detect_face_strict(self.latest_image, debug=True)
                if face_detected:
                    rospy.loginfo_throttle(2.0, "MiRo face detected (two eyes aligned).")
            rate.sleep()

if __name__ == "__main__":
    main = MiroClient()
    main.loop()
