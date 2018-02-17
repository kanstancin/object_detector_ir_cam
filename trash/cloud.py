#!/usr/bin/env python
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import cv2
import math
import rospy
from std_msgs.msg import String, Empty

class img_converter:

    def __init__(self):
        self.rate_pub = rospy.Publisher('rate', Empty)
        self.sub = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.cloud_cb)
        self.cloud = []
        self.grayDst = []

    def cloud_cb(self, msg):
        dtype_list = [(f.name, np.float32) for f in msg.fields]
        cloud_arr = np.fromstring(msg.data, dtype_list)
        
        n=960
        m=640
        self.cloud = np.reshape(cloud_arr, (n, m)) 
        cloudDst = np.zeros((n,m),np.float16)
        for i in xrange(n):
            for j in xrange(m):
            #calculates distance to the point
                cloudDst[i,j] = math.sqrt(self.cloud[i,j][0]**2+self.cloud[i,j][1]**2+self.cloud[i,j][2]**2)
                if np.isnan(cloudDst[i,j]):
                    cloudDst[i,j] = 0
        cloudDst.clip(0.15, 0.7, out=cloudDst)
        cloudDst -= 0.15
        np.multiply(cloudDst, 256, out=cloudDst, casting='unsafe')
        
        grayDst = cloudDst.astype(np.uint8)
        cv2.imshow('img2',grayDst)
        cv2.waitKey(50)
        self.rate_pub.publish()
  
if __name__ == '__main__':
      ic = img_converter()
      rospy.init_node('img_converter', anonymous=True)
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print("Shutting down")
      cv2.destroyAllWindows()
rospy.spin()
