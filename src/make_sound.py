#!/usr/bin/env python3
import os
import rospy
from std_msgs.msg import Float32MultiArray

class MiRoSound:
    def __init__(self):
        rospy.init_node("miro_sound_example")
        miro_name = os.getenv("MIRO_ROBOT_NAME")
        topic_base = f"/{miro_name}/control/sound"
        self.sound_pub = rospy.Publisher(topic_base, Float32MultiArray, queue_size=0)
        rospy.sleep(1.0)  # wait for the publisher to be ready

    def play_beep(self, freq=440.0, duration=0.5):
        """
        Play a simple beep (sine wave tone)
        freq: frequency in Hz
        duration: duration in seconds
        """
        msg = Float32MultiArray()
        msg.data = [1.0, freq, duration]  # [type=1 (tone), frequency, duration]
        self.sound_pub.publish(msg)

if __name__ == "__main__":
    sound = MiRoSound()
    rospy.sleep(1.0)
    print("Beep!")
    sound.play_beep(freq=880.0, duration=0.5)
    rospy.spin()
