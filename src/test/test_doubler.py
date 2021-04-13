#!/usr/bin/env python

import sys,os
sys.path.insert(0, os.path.abspath('..'))

import rospy
from std_msgs.msg import String
from vcu.msg import *
from random import random
from threading import Lock


data_lock = Lock()

vcu_output = VCU_Output()


def get_torque(data):
    # rospy.loginfo(rospy.get_caller_id() + "vcu.rc:%d,vcu.torque:%f", data.rc, data.tqu)
    with data_lock:
        vcu_output.torque = data.torque
        vcu_output.header = data.header


def doubler():
    rospy.init_node("doubler", anonymous=True)
    pub = rospy.Publisher("/newrizon/vcu_input", VCU_Input, queue_size=10)
    rospy.Subscriber("/newrizon/vcu_output", VCU_Output, get_torque)
    rate = rospy.Rate(1)
    rc = 0
    while not rospy.is_shutdown():
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        vcu_input = VCU_Input()
        vcu_input.header.seq = rc
        vcu_input.header.stamp = h.stamp
        vcu_input.pedal = random() * 100
        vcu_input.acceleration = random() * 10
        vcu_input.velocity = random() * 100
        rospy.loginfo(vcu_input)
        pub.publish(vcu_input)
        with data_lock:
            throttle = vcu_output.torque
            h1 = vcu_output.header
        print("rc:%d; throttle:%f" % (h1.seq, throttle))
        rc += 1
        rate.sleep()


if __name__ == "__main__":
    try:
        doubler()
    except rospy.ROSInterruptException:
        pass
