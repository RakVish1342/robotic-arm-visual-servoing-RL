#!/usr/bin/env python

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose # Same as Twist, Pose, Odom, Imu: https://answers.ros.org/question/216482/are-odom-pose-twist-and-imu-processed-differently-in-robot_localization/
import math
import numpy as np
import rospy
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, JointState
from simple_arm.srv import *


class BoxLocation(object):
    def __init__(self):
        rospy.init_node('box_location')
        print(">> Box Location node initialized")
        self.pubBoxLoc = rospy.Publisher('/simple_arm/box_location', Pose, queue_size=10)
        self.subCamera = rospy.Subscriber("/rgb_camera/image_raw", Image, self.cbCamera)
        self.image = np.zeros((480,640,3),dtype = int)
        self.cvb = CvBridge()
        self.colorThresh = 210
        self.location = (999, 999)


    def cbCamera(self, msg):
        try:
            self.image = self.cvb.imgmsg_to_cv2(msg)
        except rospy.ServiceException, e:
            rospy.logwarn("Box Location service failed: %s", e)

    def boxlocation(self):

        #time.sleep(10)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            r = np.array(self.image[:,:,0])
            temp = np.argwhere(r>=self.colorThresh)
            xsum = 0
            ysum = 0
            if(len(temp)==0):
                xloc=999
                yloc=999
            else:
                for t in temp:
                    xsum += t[1]
                    ysum += t[0]  
                xloc = xsum/len(temp)
                yloc = ysum/len(temp)

            msgOut= Pose()
            msgOut.position.x = xloc
            msgOut.position.y = yloc
            self.pubBoxLoc.publish(msgOut)    
            rate.sleep()

if __name__ == '__main__':
    try: 
        b = BoxLocation()
        b.boxlocation()
    except rospy.ROSInterruptException:
        pass
