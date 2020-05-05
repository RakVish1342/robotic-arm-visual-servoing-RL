#!/usr/bin/env python

import math
import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64

def cbJointStates(msg):
    position = msg.position
    print("PPOPOSPOSPOSPOSPOSPOSPOSPOS")
    print(position[0])
    print(position[1])
    if position[0] > 1.0 and position[1] > 1.0:
        hello = rospy.get_param("init_position_set")
        print(hello)
        rospy.set_param("init_position_set", True)
        rospy.get_param("init_position_set")
        hello = rospy.get_param("init_position_set")
        print(hello)        

def mover():
    pub_j1 = rospy.Publisher('/simple_arm/joint_1_position_controller/command',
                             Float64, queue_size=10)
    pub_j2 = rospy.Publisher('/simple_arm/joint_2_position_controller/command',
                             Float64, queue_size=10)

    sub1 = rospy.Subscriber('/simple_arm/joint_states', 
                                    JointState, cbJointStates)                             
    rospy.init_node('my_mover')
    rate = rospy.Rate(10)

    # rospy.spin()

    while (not rospy.get_param("init_position_set")):
        print("HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEY")
        pub_j1.publish(1.1)
        pub_j2.publish(1.1)

        # How is the callback still being called though when this spin() line is commented??
        # Also, why does including this rospy.spin() never allow the while loop to publish to the topic??
        # Shouldn't the callback and main body of the node run in different threads?? Why does it seem to just 
        # reach rospu.spin() and get stuck in the callback???

        # Anyway, for now since the callback is being called somehow, I will let that happen for the sake of 
        # initialization and then let the while loop run out and the node to end once initialization is done.

        # rospy.spin()
        rate.sleep()



if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass
