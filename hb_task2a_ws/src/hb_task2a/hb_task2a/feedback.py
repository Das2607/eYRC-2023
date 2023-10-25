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



# Import the required modules
##############################################################
class ArUcoDetector(Node):

    def __init__(self):
        super().__init__('feedback')

        # Accuracy
        self.iter = 0
        self.failed = 0

        # Subscribe the topic /camera/image_raw
        self.get_logger().info("Start")
        # Get cam info for calibration, in real life a chess board calibration is required
        self.sub_cam = self.create_subscription(CameraInfo, "/camera/camera_info", self.cam_callback, 10)

        # Sub for fetching images
        self.sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)

        # Publish the Pose data
        self.pub = self.create_publisher(Pose, "/detected_aruco", 10)
        self.rate = self.create_rate(1000)
        self.timer = self.create_timer(0.1 * 0.5, self.pub_callback)

        self.cv_bridge = CvBridge()

        # Load the dictionary that was used to generate the markers.
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Initialize the detector parameters using default values
        para =  cv2.aruco.DetectorParameters()

        # Better Refinement
        para.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        para.aprilTagDeglitch = 1
        para.maxErroneousBitsInBorderRate = 0.35
        para.errorCorrectionRate = 1                
        para.polygonalApproxAccuracyRate = 0.090
        self.detector = cv2.aruco.ArucoDetector(dictionary, para)

        # Pose msg to be published
        self.pose = Pose()


    def cam_callback(self, info):
        # Gets calib values from gazebo
        self.dist = np.array(info.d)
        self.mtx = np.array(info.k).reshape(3,3)

    def pub_callback(self):
        self.pub.publish(self.pose)

    def remove_camera_dist(self, cv_image_raw):
        # Remove Camera distortion
        h, w = cv_image_raw.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(cv_image_raw, self.mtx, self.dist, None, newcameramtx)

        x, y, w_, h_ = roi
        cv_image_raw = dst[y : y + h_, x : x + w_]

        # Sharpen the image 
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        srp = cv2.filter2D(cv_image_raw, -1, kernel)

        # Make the image into grayscale for better thresholding
        gray = cv2.cvtColor(srp, cv2.COLOR_BGR2GRAY)

        #Thresholding to ensure consistent detection of aruco
        ret, cv_image = cv2.threshold(gray, 110, 225, cv2.THRESH_BINARY)
         
        #Detect the aruco marker
        corners, ids, _ = self.detector.detectMarkers(cv_image)

        #Find the corners of 4 stationary aruco marker (board corners)
        try:
            for i in range(5):
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
            imgOut = cv2.warpPerspective(cv_image, matrix, (cv_image_raw.shape[1], cv_image_raw.shape[0]))

            #Threshold again to filter gray pixels after fixing perspective
            ret, imgOut = cv2.threshold(imgOut, 80, 255, cv2.THRESH_BINARY)

        except:
            # If an error is encountered, return the original image after thresholding
            imgOut = cv_image

        return imgOut


    def image_callback(self, msg):
        self.iter += 1
        #convert ROS image to opencv image
        cv_image_raw = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 

        try:
            _ = (self.mtx, self.dist)
        except:
            return
        
        imgOut = self.remove_camera_dist(cv_image_raw)

        # Detect again after the image is flattened
        corners, ids, _ = self.detector.detectMarkers(imgOut)
        try:
            for i in range(0, len(ids)):

                # Scan only for the e_bot (e_bot has id as 1)
                if (ids[i] == 1):
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.69, self.mtx, self.dist)
                    self.pose.position.x = float(tvec.flatten()[0])
                    self.pose.position.y = -float(tvec.flatten()[1])
                    self.pose.orientation.z = -float(euler_from_matrix(cv2.Rodrigues(rvec)[0])[2])
        except:
            print("Error: 185")

        # Checks for all aruco markers
        try:
            assert(len(ids) == 5)
            self.failed += 1
        except:
            print(ids)
            ...
        # cv2.imshow("test",imgOut)
        # cv2.imshow("test2",cv_image_raw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if (self.iter % 100 == 0):
            self.get_logger().info(f'{self.iter} >>> {self.failed/self.iter * 100}')
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