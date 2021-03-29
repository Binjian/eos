#!/usr/bin/env python


## Simple talker demo that listens to std_msgs/Strings published
## to the 'chatter' topic

import rospy
from std_msgs.msg import String
from vcu.msg import *
from threading import Lock

data_lock = Lock()


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


def get_torque(data):
    rospy.loginfo(rospy.get_caller_id() + "vcu.rc:%d,vcu.torque:%f", data.rc, data.tqu)
    with data_lock:
        vcu_output.rc = data.rc
        vcu_output.torque = data.tqu


def callback(data):
    rospy.loginfo(
        rospy.get_caller_id() + "vcu.rc:%d,vcu.ped:%f,vcu.acc:%f,vcu.vel:%f",
        data.rc,
        data.ped,
        data.acc,
        data.vel,
    )


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node("listener", anonymous=True)

    rospy.Subscriber("vcu", VCU_Input, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    listener()
