#! /usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		Hologlyph Bots (HB) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2A of Hologlyph Bots (HB) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
# Team ID:		[ 2865 ]
# Author List:	[Shreyas Das]
# Filename:		controller.py
# Functions:
#			[ Comma separated list of functions in this file ]
# Nodes:		Add your publishing and subscribing node
################### IMPORT MODULES #######################
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
import cv2 
from cv_bridge import CvBridge
import math
import numpy as np
from tf_transformations import euler_from_matrix
##############################################################

class MovingAverage():
    def __init__(self):
        self.que = []
        self.avg = [0.0, 0.0, 0.0]
        self.init = False
        self.thres = 6

    def insert(self, lst = [0, 0, 0]):
        assert(len(lst) == 3)
        self.init = True
        self.que.append(lst)
        for i in range(len(lst)):
            self.avg[i] += lst[i]

        if len(self.que) > self.thres:
            rm = self.que.pop(0)
            for i in range(len(lst)):
                self.avg[i] -= rm[i]

    def get_avg(self):
        if not self.init:
            return self.avg
        avg = round(self.avg[0]/len(self.que), 3), round(self.avg[1]/len(self.que), 3), round(self.avg[2]/len(self.que), 3)
        return avg
            

class ArUcoDetector(Node):

    def __init__(self):
        super().__init__('feedback')

        # Accuracy of detection
        self.iter = 0
        self.failed = 0

        # Subscribe the topic /camera/image_raw
        self.get_logger().info("Start")
        # Get cam info for calibration, in real life a chess board calibration is required
        self.sub_cam = self.create_subscription(CameraInfo, "/camera/camera_info", self.cam_callback, 10)

        # Sub for fetching images
        self.sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)

        # Publish the Pose data
        self.pub_1 = self.create_publisher(Pose, "/detected_aruco_1", 10)
        self.pub_2 = self.create_publisher(Pose, "/detected_aruco_2", 10)
        self.pub_3 = self.create_publisher(Pose, "/detected_aruco_3", 10)
        self.rate = self.create_rate(1000)
        self.timer = self.create_timer(0.1 * 0.1, self.pub_callback)

        self.cv_bridge = CvBridge()

        # Load the dictionary that was used to generate the markers.
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Initialize the detector parameters using default values
        para =  cv2.aruco.DetectorParameters()

        # Better Refinement
        para.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        para.aprilTagDeglitch = 1
        para.maxErroneousBitsInBorderRate = 0.35
        para.errorCorrectionRate = 1                
        para.polygonalApproxAccuracyRate = 0.090
        # Make a ArUcoDetector
        self.detector = cv2.aruco.ArucoDetector(dictionary, para)

        # Pose msg to be published
        self.pose_1 = Pose()
        self.pose_2 = Pose()
        self.pose_3 = Pose()

        self.pose_arr = [self.pose_1, self.pose_2, self.pose_3]

        # Scale for convert units to Gazebo Units
        self.scale = 0.01

        # Size matrix of a marker (measured in blender)
        marker_size = 70.3463
        self.marker_points = np.array([
            [-marker_size / 2, +marker_size / 2, 0],
            [+marker_size / 2, +marker_size / 2, 0],
            [+marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]],
        dtype=np.float32)


        # rvec and tvec definition in global namespace, used for refining pose every iteration
        self.rvec = [0,0,0]
        self.tvec = [0,0,0]           
        self.vecs_init = False

        # Moving Average to refine pose jitter
        # TODO: Add EKF instead of Moving Average
        self.moving_avg = [MovingAverage(), MovingAverage(), MovingAverage()]

        # [DEBUG ONLY]
        # self.thres_bw = 0


    def cam_callback(self, info):
        # Gets calib values from gazebo
        self.dist = np.array(info.d)
        self.mtx = np.array(info.k).reshape(3,3)

    def pub_callback(self):
        self.pub_1.publish(self.pose_1)
        self.pub_2.publish(self.pose_2)
        self.pub_3.publish(self.pose_3)
        # print(self.pose_3.position.x,self.pose_3.position.y)

    def remove_camera_dist(self, cv_image_raw_):
        # Remove Camera distortion
        h, w = cv_image_raw_.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(cv_image_raw_, self.mtx, self.dist, None, newcameramtx)

        x, y, w_, h_ = roi
        cv_image_raw = dst[y : y + h_, x : x + w_] 

        # Sharpen the image 
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        srp = cv2.filter2D(cv_image_raw, -1, kernel)

        # Make the image into grayscale for better thresholding
        gray = cv2.cvtColor(srp, cv2.COLOR_BGR2GRAY)

        # Blur to remove grid lines
        blur = cv2.blur(gray,(3,3))

        #Thresholding to ensure consistent detection of aruco
        ret, cv_image = cv2.threshold(blur, 77.5, 225, cv2.THRESH_BINARY)
         
        #Detect the aruco marker
        corners, ids, _ = self.detector.detectMarkers(cv_image)

        #Find the corners of 4 stationary aruco marker (board corners)
        try:
            for i in range(len(ids)):
                if (ids[i] == 8):
                    tl = corners[i][0][0]
                if (ids[i] == 10):
                    tr = corners[i][0][1]
                if (ids[i] == 12):
                    br = corners[i][0][2]
                if (ids[i] == 4):
                    bl = corners[i][0][3]


            # Find Perspective tilt and rescale, skew, rotate it
            pts1 = np.array([tl, tr, br, bl])
            pts2 = np.float32([[5,5], [w-5,5], [w-5,h-5], [5,h-5]])
            matrix, _ = cv2.findHomography(pts1, pts2)
            imgOut = cv2.warpPerspective(gray, matrix, (cv_image_raw.shape[1], cv_image_raw.shape[0]))

            #Threshold again to filter gray pixels after fixing perspective
            imgOut = cv2.blur(imgOut,(2,2))
            ret, imgOut = cv2.threshold(imgOut, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        except Exception as e:
            # If an error is encountered, return the original image after thresholding
            print(f"E{e}")
            imgOut = gray
            imgOut = cv2.blur(gray,(2,2))
            ret, imgOut = cv2.threshold(imgOut, 110, 255, cv2.THRESH_BINARY)

        # For debugging
        # cv2.aruco.drawDetectedMarkers(cv_image, corners)
        # cv2.imshow("test2",cv_image)
        return imgOut

    def image_callback(self, msg):
        self.iter += 1
        #convert ROS image to opencv image
        cv_image_raw = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 

        # Ensures image_callback runs only when mtx and dist are present
        try:
            _ = (self.mtx, self.dist)
        except:
            return
        
        imgOut = self.remove_camera_dist(cv_image_raw)

        # Detect again after fixing the perspective
        corners, ids, _ = self.detector.detectMarkers(imgOut)
        try:
            for i in range(0, len(ids)):
                # Scan only for the e_bots (3 e_bots have id as 1, 2, 3)
                if (ids[i] <= len(self.pose_arr)):
                    #Old pose detect
                    # self.rvec[int(ids[i]-1)], self.tvec[int(ids[i]-1)], markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 70.3463, self.mtx, self.dist)

                    # Newer pose detect
                    inliers = 0
                    if self.vecs_init:
                        _, self.rvec[int(ids[i]-1)], self.tvec[int(ids[i]-1)]= cv2.solvePnP(self.marker_points, corners[i], self.mtx, self.dist, self.rvec[int(ids[i]-1)], self.tvec[int(ids[i]-1)], True, flags =  cv2.SOLVEPNP_ITERATIVE )
                    else:
                        # self.get_logger().info(f"Found Bot {int(ids[i])}")
                        _, self.rvec[int(ids[i]-1)], self.tvec[int(ids[i]-1)] = cv2.solvePnP(self.marker_points, corners[i], self.mtx, self.dist, useExtrinsicGuess = False, flags=cv2.SOLVEPNP_IPPE_SQUARE)

                    # Add them to be published
                    x = round(float(self.tvec[int(ids[i]-1)].flatten()[0]) * self.scale, 3) 
                    y = -round(float(self.tvec[int(ids[i]-1)].flatten()[1]) * self.scale, 3)  
                    z = -round(float(euler_from_matrix(cv2.Rodrigues(self.rvec[int(ids[i])-1])[0])[2]), 3)
                    temp_arr = [x, y, z]
                    # if (int(ids[i]-1) == 1):
                    #     print(self.moving_avg[int(ids[i]-1)].get_avg())
                    self.moving_avg[int(ids[i]-1)].insert(temp_arr)
                    
                    self.pose_arr[int(ids[i]-1)].position.x = self.moving_avg[int(ids[i]-1)].get_avg()[0]
                    self.pose_arr[int(ids[i]-1)].position.y = self.moving_avg[int(ids[i]-1)].get_avg()[1]
                    self.pose_arr[int(ids[i]-1)].orientation.z = self.moving_avg[int(ids[i]-1)].get_avg()[2]
                    

        except Exception as e:
            print(f"Error: 185 {e}")

        # Checks for all aruco markers
        try:
            temp = 0
            for k in range(0, len(ids)):
                if (ids[k] <= len(self.pose_arr)):
                    temp += 1
            assert(temp == 3)
            self.vecs_init = True
            self.failed += 1
        except:
            ...

        # For debugging
        # cv2.aruco.drawDetectedMarkers(imgOut, corners)
        # cv2.imshow("test",cv2.cvtColor(imgOut, cv2.COLOR_BGR2RGB))
        # cv2.imshow("test2",cv_image_raw)
        # cv2.waitKey(1)
        if (self.iter % 100 == 0):
            self.get_logger().info(f'Detection Rate >>> {self.failed/self.iter * 100}')
            self.iter = 0
            self.failed = 0

        
        

       


def main(args=None):
    rclpy.init(args=args)

    aruco_detector = ArUcoDetector()

    rclpy.spin(aruco_detector)

    aruco_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



"""
Board Corner Markers
+++++++++++++++++++++++++++++++
810
412
-------------------------------
[10]
[[[913.  37.]
[964.  37.]
[964.  88.]
[913.  88.]]]
-------------------------------
[8]
[[[34.000004 38.      ]
[85.       38.      ]
[85.       89.      ]
[33.999996 89.      ]]]
-------------------------------
[12]
[[[912.97424 916.0261 ]
[964.      915.9878 ]
[964.      966.     ]
[914.1551  966.     ]]]
-------------------------------
[1]
[[[523.4341  515.29553]
[494.84586 513.8977 ]
[496.28326 485.7005 ]
[524.25165 486.84137]]]
-------------------------------
[4]
[[[ 35.000004 916.      ]
[ 86.       916.      ]
[ 86.       967.      ]
[ 34.999996 967.      ]]]
-------------------------------
"""