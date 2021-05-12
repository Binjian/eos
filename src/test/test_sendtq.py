#!/usr/bin/env python

import rospy
import std_msgs.msg
from vcu.msg import *
from random import random

# vcu_output = VCU_Output()
def talker():
    pub = rospy.Publisher("vcuo", VCU_Output, queue_size=10)
    rospy.init_node("talker", anonymous=True)
    rate = rospy.Rate(1)
    rc = 0
    while not rospy.is_shutdown():
        vcu_output = VCU_Output()
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()

        vcu_output.header.seq = rc
        vcu_output.header.stamp = h.stamp
        vcu_output.rc = rc
        vcu_output.tqu = random() * 100
        rospy.loginfo(vcu_output)
        pub.publish(vcu_output)
        rc += 1
        rate.sleep()


if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
