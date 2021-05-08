import rospy
import std_msgs.msg
import sys, os

sys.path.insert(0, os.path.abspath(".."))

from comm.vcu.msg import *
from threading import Lock

# from demo import data_lock, vcu_output


def talker(pub, rc, vel, acc, ped):
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    vcu_input1 = VCU_Input()
    vcu_input1.header.seq = rc
    vcu_input1.header.stamp = h.stamp
    vcu_input1.pedal = ped * 100
    vcu_input1.acceleration = acc
    vcu_input1.velocity = vel
    # rospy.loginfo(vcu_input1)
    pub.publish(vcu_input1)
