#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('realsense_camera')
import sys
import rospy
import cv2
import numpy as np
import time
import math
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
from sensor_msgs.msg import PointCloud2, PointField

def ifBolt(m2,img):
    m=0
    for i in grip[m2]:
        best_I, best_J = i
        dst = dist(lines[best_I][0][0],lines[best_I][0][1],coeff[best_I][0], \
                                       coeff[best_J][0],coeff[best_J][1])
        length = 0.7*max(coeff[best_I][2],coeff[best_J][2])
        x0 = float(coeff[best_J][3]+coeff[best_I][3])/2.0
        y0 = float(coeff[best_J][4]+coeff[best_I][4])/2.0
        k = float(coeff[best_J][0]+coeff[best_I][0])/2.0

        cos = (float(length)/float(math.sqrt(1+math.pow(float(k),2))))
        x1 = cos + x0
        y1 = cos*k + y0
        stepX = float(x1 - x0)/10.0
        stepY = float(y1 - y0)/10.0
        if k==0: k = 0.01
        deltaX = float(float(0.7*dst))/float(math.sqrt(1+math.pow(1.0/float(k),2)))
        deltaY = deltaX*float(1.0/float(k))
        for o in range(0,22):
            #cls[int(y1+deltaY),int(x1-deltaX)] = 255
            #cls[int(y1-deltaY),int(x1+deltaX)] = 255
            if (cls[int(y1+deltaY),int(x1-deltaX)]!=0) and ( \
                                cls[int(y1-deltaY),int(x1+deltaX)]!=0):
                if ((cls[int(y1-3*stepY),int(x1-3*stepX)]!=0) != ( \
                                cls[int(y1+3*stepY),int(x1+3*stepX)]!=0)):
                
                    cls[int(y1-3*stepY),int(x1-3*stepX)] = 255
                    cls[int(y1+3*stepY),int(x1+3*stepX)] = 255
                    cls[int(y1+deltaY),int(x1-deltaX)] = 255
                    cls[int(y1-deltaY),int(x1+deltaX)] = 255
                    cv2.circle(img,(int(x1),int(y1)),10,(255),-1)
                    m = 1       
            x1 = x1 - stepX
            y1 = y1 - stepY
    return m

def gripper(line_i, line_j,j):
    x0 = float(coeff[line_j][3]+coeff[line_i][3])/2.0
    y0 = float(coeff[line_j][4]+coeff[line_i][4])/2.0
    k = float(coeff[line_i][0]+coeff[line_j][0])/2.0
    b = y0 - x0*k
    cos = (float(30)/float(math.sqrt(1+math.pow((coeff[line_j][0]+coeff[line_i][0])/2,2))))
    x1 = cos + x0
    y1 = cos*k+y0
    x2 = x0 - cos
    y2 = y0 - cos * k
    x0,y0,x1,y1,x2,y2 = [int(x0),int(y0),int(x1),int(y1),int(x2),int(y2)]
    cv2.rectangle(img, (x0-20,y0-20),(x0+20,y0+20),(0,255,0),3)
    cv2.line(img,(x1,y1),(x2,y2),(0,255,255),3)
    cv2.putText(img, metr[j],(x0-20,y0-25),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
    
def parallel(line_i, a, val):
    #global lines,coeff,dilation
    maxDist = 0
    for i in xrange(lines.shape[0]):
        if (abs((math.atan2(coeff[i,5],coeff[i,6])-math.atan2(coeff[line_i,5],coeff[line_i,6]))) < 0.25) and ( \
            i!=line_i) and (lines[i,0,0]!=0) and (val == dilation[lines[i,0,1],lines[i,0,0]]):
            dst = dist(lines[line_i,0,0],lines[line_i,0,1],coeff[line_i,0],coeff[i,0],coeff[i,1])
            if (dst >= maxDist) and (float(min(coeff[i,2],coeff[line_i,2]))/float( \
                                max(coeff[i,2],coeff[line_i,2])) > 0.5):
                maxDist = dst    
                best_I = i
    return best_I

def dist(x0, y0, k0, k1, b1):
    if k0!=0:
        k2 = float(-1.0/k0)
    else: k2 = -337
    b2 = y0 - k2*x0
    x = (float(b2 - b1))/(float(k1 - k2))
    y = k1*x + b1
    dst = math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0, 2))
    return dst

def line(gray): 
    k = 25
    
    edges = cv2.Canny(gray, k, 70)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations= 3)
    kernel2 = np.ones((9,9),np.uint8)
    cls = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2, iterations= 3)
    
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10,
                param2=17, minRadius=5, maxRadius=25)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations= 4)
    if circles is not None:
        for (x,y,r) in circles[0,:]:
            cv2.circle(closing, (x, y), r, 255 , -1)
            cv2.circle(dilation, (x, y), r, 255 , -1)
            cv2.circle(cls, (x, y), r, 255 , -1)
            
    grad = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    outer = cv2.morphologyEx(grad, cv2.MORPH_DILATE, kernel, iterations = 2)
    
    h, w = gray.shape[:2]
    mask = np.zeros((h+2,w+2), np.uint8)
                           
    for i in range(0, h, 3):
        if (outer[i,0] == 255):
            cv2.floodFill(outer,mask,(0,i),0)
        if (outer[i,w-1] == 255):
            cv2.floodFill(outer,mask,(w-1,i),0)
    for j in range(0, w, 3):
        if (outer[0,j] == 255):
            cv2.floodFill(outer,mask,(j,0),0)
        if (outer[h-1,j] == 255):
            cv2.floodFill(outer,mask,(j,h-1),0)
            
    img2 = cv2.bitwise_and(edges,edges,mask = outer)
    lines = cv2.HoughLinesP(img2,1,np.pi/180, 17,np.array([]), 13,7) 
    '''if lines is not None:
        for i in xrange(lines.shape[0]):#
            cv2.line(zeros, (lines[i,0,0], lines[i,0,1]), (lines[i,0,2], lines[i,0,3]), \
                         (255),1,cv2.LINE_AA)  
    l2 = cv2.HoughLinesP(zeros,1,np.pi/180, 17,np.array([]), 13,7)  
    if l2 is not None:
        lines = np.concatenate((lines,l2))''' 
 
    for i in range(1,7):
        k = 15*i + k
        edges = cv2.Canny(gray, k, k * 2)
        img2 = cv2.bitwise_and(edges,edges,mask = outer)
        l2 = cv2.HoughLinesP(img2,1,np.pi/180, 17,np.array([]), 13,7)
        if l2 is not None:
            lines = np.concatenate((lines,l2))   
    return lines, outer,circles,edges,dilation,cls


def display(image, display_min, display_max):
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
    self.rate_pub = rospy.Publisher('rate', Empty)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/ir/image_raw",Image,self.callback)
    self.sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.cloud_cb)
    self.h = []
    self.cloud = []
    self.grayDst = []
   
  def cloud_cb(self, msg):
    dtype_list = [(f.name, np.float32) for f in msg.fields]
    cloud_arr = np.fromstring(msg.data, dtype_list)
    #cloud_arr = cloud_arr[0:307200]
    print(cloud_arr)
    print(cloud_arr.shape)
    n=640
    m=640
    self.cloud = np.reshape(cloud_arr, (n, m)) 
    self.cloud = self.cloud[:,:]
    lut = np.arange(2**16, dtype='uint16')
    cloudDst = np.zeros((n,m),np.float16)
    for i in xrange(n):
        for j in xrange(m):
           
            cloudDst[i,j] = math.sqrt(self.cloud[i,j][0]**2+self.cloud[i,j][2]**2+self.cloud[i,j][1]**2)
            if np.isnan(cloudDst[i,j]):
                cloudDst[i,j] = 0
                
    print(np.average(cloudDst))
    cloudDst.clip(0.15, 1.5, out=cloudDst)
    cloudDst -= 0.15
    np.multiply(cloudDst, 256,
                    out=cloudDst, casting='unsafe')
    
    self.grayDst = cloudDst.astype(np.uint8)
    self.rate_pub.publish()
    
  def dstPlane(self, angle):
    "angle between Z of the cam frame and normale of the plane"
    self.cloud = self.cloud[160:480,:]
    print(self.cloud.shape)
    h1=[]
    #print(self.cloud[300,500])
    for i in xrange(0, 480, 3):
        for j in xrange(0, 640, 3):
            dist = math.sqrt(self.cloud[i,j][1] ** 2 + self.cloud[i,j][2] ** 2)
            theta = math.degrees(math.acos(self.cloud[i,j][2]/dist))
            rho = angle - theta
            h1.append(dist * math.cos(math.radians(rho)))
    self.h = np.array(h1) 
            
    #theta = math.acos(z/dist) 
    #rho = angle - theta
    #h = dist * math.cos(rho)            
    '''for i in np.nditer(np.nonzero(dilation)])):

    for i in xrange(0, h, 3):
        for j in xrange(0, w, 3):
            if dilation[i,j] == 0:
                _=0
                # print(self.cloud[j,i])''' 
                     
  def callback(self,data):
    try:
      ir = self.bridge.imgmsg_to_cv2(data)
    except CvBridgeError as e:
      print(e)
      
    global lines,coeff,dilation, outer, circles, edges, cls 
    global grip,img,metr,zeros,gray,cloud
    
    mtxloaded = np.load('mtx.npy')
    distloaded = np.load('dist.npy')
    display_min = 1
    display_max = 30000
    
    img = lut_display(ir,display_min,display_max)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtxloaded,distloaded,(w,h),1,(w,h))
    zeros = np.zeros((h,w),np.uint8)
    img = cv2.undistort(img, mtxloaded, distloaded, None, newcameramtx)
  
    x,y,w,h = roi
    img = img[y:y+h, x:x+w]
   
    ##########################
    metric = np.zeros((3,5),np.float16)
    metric[0] = [3.1,3.5,3.9,2.7,4.3]    # BOLT
    metric[1] = [100,0.7,0.55,-1,-1]     # NUT
    metric[2] = [3.6, 4.0, 3.2,4.4,-1]   # BALK
    metr = ['BOLT','NUT','BALK']

    met_count = np.zeros((3),np.uint16)
    met_count[0] = 1
    met_count[1] = 2
    met_count[2] = 1
    met = []
    for i in range(3):
        met.append([])
    pairs = []
    grip = []
    for i in range(15):
        grip.append([])
    met_circ = []
    for i in range(3):
        met_circ.append([])
    
    gray = img
    gray = cv2.medianBlur(gray, 3)
    h, w = gray.shape[:2]
    #self.dstPlane(0)
    x,y=[400,200]
    #cv2.circle(img,(x,y),7,255,-1)
    #print(self.cloud[y-30,x+20])
    lines, outer, circles, edges, dilation,cls = line(gray)
    if lines is not(None) and 1==0:
        a = len(lines)

        mask = np.zeros((h+2,w+2), np.uint8)

        coeff = []
        for i in xrange(lines.shape[0]):#
            x1,y1,x2,y2 = lines.item(i,0,0),lines.item(i,0,1),lines.item(i,0,2 \
                                ), lines.item(i,0,3)                  
            lenght = int(math.sqrt((x1-x2)**2+(y1-y2)**2))
            if x1 != x2:
                k = round((float(y2 - y1))/(float(x2 - x1)),5)
                b = y1 - k*x1
            else:
                k = -337
                b = y1 - k*x1
            coeff.append([k,b,lenght,(x1+x2)/2,(y1+y2)/2,y2-y1,x2-x1])
        coeff = np.array(coeff)
        
        best_I, best_J, maxDist = 0,0,0
        k=0
        ###   removin' close lines   ###
        for i in xrange(lines.shape[0]):
            if (lines[i,0,0] != 0):
                for j in xrange(lines.shape[0]):
                    if (lines[j,0,0] != 0) and (i != j):
                        cdst = math.sqrt(math.pow(coeff[j,3]-coeff[i,3],2) + math.pow(coeff[j,4]-coeff[i,4],2))
                        if (abs((math.atan2(coeff[i,5],coeff[i,6])-math.atan2(coeff[j,5],coeff[j,6]))) < 0.2) and       (cdst < 100):
                            dst = dist(lines[i,0,0],lines[i,0,1],coeff[i,0],coeff[j,0],coeff[j,1])
                            if (coeff[j,2] < coeff[i,2]) and (dst<6):
                                lines[j]=0
                                k += 1
        ################################

        ###  showin' remaining lines ###
        for i in xrange(lines.shape[0]):#
            cv2.line(img, (lines[i,0,0], lines[i,0,1]), (lines[i,0,2], lines[i,0,3]), \
                     (0,255,0),1,cv2.LINE_AA)
            
        ################################

        ### findin' parallel line   ####
        m=1
        a = len(lines)

        for i in xrange(lines.shape[0]):#
            best_J=-1
            maxDist = 0
            d = dilation.item(lines[i,0,1],lines[i,0,0])
            if  (lines[i,0,0]!=0):
                for j in xrange(lines.shape[0]):
                    if (lines[j,0,0]!=0) and (abs((math.atan2(coeff[i,5],coeff[i,6])- \
                                        math.atan2(coeff[j,5],coeff[j,6]))) < 0.25) and (i!=j):
                        if (dilation[lines[j,0,1],lines[j,0,0]] == 255):
                            cv2.floodFill(dilation,mask,(lines[j,0,0],lines[j,0,1]),m*10+100)
                            m=m+1
                        if (dilation[lines[j,0,1],lines[j,0,0]] == dilation[lines[i,0,1],lines[i,0,0]]):
                            dst = dist(lines[i,0,0],lines[i,0,1],coeff[i,0],coeff[j,0],coeff[j,1])
                            if (dst >= maxDist) and (float(min(coeff[i,2],coeff[j,2]))/float( \
                                max(coeff[i,2],coeff[j,2])) > 0.5):
                                maxDist = dst              
                                best_J = j
            
            if (best_J != -1):
                
                c = 2
                d = dilation[lines[best_J,0,1],lines[best_J,0,0]]
                best_I = parallel(best_J, a, d)
                
                length = (coeff[best_I,2]+coeff[best_J,2])/2
                dst = dist(lines[best_I,0,0],lines[best_I,0,1],coeff[best_I,0],coeff[best_J,0],coeff[best_J,1])
                
                cdst = math.sqrt(math.pow(coeff[best_J,3]-coeff[best_I,3],2) + math.pow(coeff[best_J,4]-coeff[best_I,4],2))
                if (dst != 0):
                    
                    if ((float(min(dst, cdst)))/(float(max(dst, cdst))) > 0.5):
                        
                        cv2.line(img, (lines[best_J,0,0], lines[best_J,0,1]), \
                                     (lines[best_J,0,2], lines[best_J,0,3]), \
                                     (0,0,255),1,cv2.LINE_AA)
                        cv2.line(img, (lines[best_I,0,0], lines[best_I,0,1]), \
                                     (lines[best_I,0,2], lines[best_I,0,3]), \
                                     (0,0,255),1,cv2.LINE_AA)
                        if best_I not in pairs:
                            dst = dist(lines[best_I,0,0],lines[best_I,0,1],coeff[best_I,0],coeff[best_J,0],coeff[best_J,1])
                            m2 = (d-100)/10
                            if dst==0:dst=0.001
                            coef = float(max(coeff[best_J,2],coeff[best_I,2]))/float(dst)
                           
                            k = np.where(abs(metric - coef) <= 0.2)
                            
                            for i in range(len(k[0])):
                                grip[m2].append([best_I,best_J])
                                met[k[0][i]].append(m2)
                                
                                cv2.line(img, (lines[best_J,0,0], lines[best_J,0,1]), \
                                     (lines[best_J,0,2], lines[best_J,0,3]), \
                                     (255),3,cv2.LINE_AA)
                                
                                cv2.line(img, (lines[best_I,0,0], lines[best_I,0,1]), \
                                     (lines[best_I,0,2], lines[best_I,0,3]), \
                                     (255),3,cv2.LINE_AA)
                                
                        if best_I not in pairs:
                            pairs.append(best_I)
                        if best_J not in pairs:
                            pairs.append(best_J)
                              
        
        if circles is not None:
            for (x,y,r) in circles[0,:]:
                d = dilation[int(y),int(x)]
                m2 = (d-100)/10
                k = np.where(metric == 100)
                if d > 0:
                    for i in range(len(k[0])):
                        met_circ[k[0][i]].append(m2)
                        
                        cv2.circle(edges, (x, y), r, 255 , 2)
               
        c = 0
        c0 = 0
        for i in range(1,m):
            bolt = ifBolt(i,img)
            for j in range(len(metric)):
                for z in range(len(met[j])):
                    if met[j][z] == i:
                        c += 1
                for l in range(len(met_circ[j])):   
                    if met_circ[j][l] == i:
                        c0 = 1
                if c >= met_count[j] and (metric[j][0]==100 and c0==1 or
                        metric[j][0]!=100 and c0==0):
                    if (metr[j] == 'BOLT'):
                        if (bolt==1):
                            print(str(i)+' - '+metr[j])
                            for o in range(0, len(grip[i])):
                                gripper(grip[i][o][0],grip[i][o][1],j)
                    elif (metr[j] == 'BALK'):
                        if (bolt==0):
                            print(str(i)+' - '+metr[j])
                            for o in range(0, len(grip[i])):
                                gripper(grip[i][o][0],grip[i][o][1],j)
                            
                    else:    
                        print(str(i)+' - '+metr[j])
                        for o in range(0, len(grip[i])):
                            gripper(grip[i][o][0],grip[i][o][1],j)
                        
                c = 0
                c0 = 0
    #print(np.nanmean(self.h))

    cv2.imshow('img',img)
    cv2.imshow('img2',self.grayDst)
    #cv2.imshow('edges',edges)
    #cv2.imshow('cls',cls)

    cv2.waitKey(3)

    #print(met)
    
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(ir))
    except CvBridgeError as e:
      print(e)

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
