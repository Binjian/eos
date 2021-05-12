#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from vcu.msg import *
from random import random


def talker():
    pub = rospy.Publisher("vcu", VCU_Input, queue_size=10)
    rospy.init_node("talker", anonymous=True)
    rate = rospy.Rate(10)
    rc = 0
    while not rospy.is_shutdown():
        vcu_input = VCU_Input()
        vcu_input.rc = rc
        vcu_input.ped = random()
        vcu_input.acc = random() * 10
        vcu_input.vel = random() * 100
        rospy.loginfo(vcu_input)
        pub.publish(vcu_input)
        rc += 1
        rate.sleep()


vcu_output = VCU_Output()

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
