#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('realsense_camera')
import sys
import rospy
import cv2
import numpy as np
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt

import sensor_msgs.point_cloud2 as pc2

def display(image, display_min, display_max): # copied from Bi Rico
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    np.floor_divide(image, (display_max - display_min + 1) / 256,
                    out=image, casting='unsafe')
    return image.astype(np.uint8)

def lut_display(image, display_min, display_max) :
    lut = np.arange(2**16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 65280.0) ** invGamma) * 255
        for i in np.arange(0,65536)]).astype("uint16")   
    return np.take(table,image).astype("uint8")
    
table = np.array([((i / 65280.0) ** 0.45) * 255
            for i in np.arange(0,65536)]).astype("uint16")
             
class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/depth/points/PointCloud2",Image,self.callback)

  def callback(self,data):
    for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
        print( " x : %f  y: %f  z: %f" %(p[0],p[1],p[2]))
      
    print('g')
    
   

def main(args):
  
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
