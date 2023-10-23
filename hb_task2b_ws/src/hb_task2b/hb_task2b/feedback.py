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


    def cam_callback(self, info):
        # Gets calib values from gazebo
        self.dist = np.array(info.d)
        self.mtx = np.array(info.k).reshape(3,3)

    def pub_callback(self):
        self.pub_1.publish(self.pose_1)
        self.pub_2.publish(self.pose_2)
        self.pub_3.publish(self.pose_3)


    def image_callback(self, msg):
        self.iter += 1
        #convert ROS image to opencv image
        cv_image_raw = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 

        # Ensures image_callback runs only when mtx and dist are present
        try:
            temp = (self.mtx, self.dist)
        except:
            return
        
        # Remove Camera distortion

        h, w = cv_image_raw.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(cv_image_raw, self.mtx, self.dist, None, newcameramtx)

        x, y, w, h = roi
        cv_image_raw = dst[y : y + h, x : x + w]
        h, w = cv_image_raw.shape[:2]

        # Sharpen the image 
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        srp = cv2.filter2D(cv_image_raw, -1, kernel)

        # Make the image into grayscale for better thresholding
        gray = cv2.cvtColor(srp, cv2.COLOR_BGR2GRAY)

        #Thresholding to ensure consistent detection of aruco
        ret, cv_image = cv2.threshold(gray, 110, 225, cv2.THRESH_BINARY)
         
        #Detect the aruco marker
        corners, ids, _ = self.detector.detectMarkers(cv_image)
        try:
            # assert(1==2)
            for i in range(5):
                if (ids[i] == 8):
                    tl = corners[i][0][0]
                if (ids[i] == 10):
                    tr = corners[i][0][1]
                if (ids[i] == 12):
                    br = corners[i][0][2]
                if (ids[i] == 4):
                    bl = corners[i][0][3]

            pts1 = np.array([tl, tr, br, bl])
            pts2 = np.float32([[5,5], [w-5,5], [w-5,h-5], [5,h-5]])
            matrix, _ = cv2.findHomography(pts1, pts2)
            imgOut = cv2.warpPerspective(cv_image, matrix, (cv_image_raw.shape[1], cv_image_raw.shape[0]))
            ret, imgOut = cv2.threshold(imgOut, 110, 255, cv2.THRESH_BINARY)
        except:
            imgOut = cv_image
        """
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

        # detect again after the image is flattened
        corners, ids, _ = self.detector.detectMarkers(imgOut)
        try:
            cv2.aruco.drawDetectedMarkers(imgOut, corners, ids)
            for i in range(0, len(ids)):
                # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.mtx, self.dist)
                # cv2.drawFrameAxes(imgOut, self.mtx, self.dist, rvec, tvec, 0.01)

                # Scan only for the e_bot (e_bot has id as 1,2,3)
                for j in range(len(self.pose_arr)):
                    if (ids[i] == (j+1)):
                        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.69, self.mtx, self.dist)
                        self.pose_arr[j].position.x = float(tvec.flatten()[0])
                        self.pose_arr[j].position.y = -float(tvec.flatten()[1])
                        self.pose_arr[j].orientation.z = -float(euler_from_matrix(cv2.Rodrigues(rvec)[0])[2])
                    # print(self.pose.position.x, self.pose.position.y, self.pose.orientation.z)
        except Exception as e:
            print(f"Error: 185 {e}")

        # Checks for all aruco markers
        try:
            assert(len(ids) == 7)
            self.failed += 1
        except:
            ...
            print("[ ERROR ]")
            # print(ids)
        # cv2.imshow("test",imgOut)
        # cv2.imshow("test2",cv_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if (self.iter % 100 == 0):
            self.get_logger().info(f'{self.iter} >>> {self.failed/self.iter * 100}')

        
        

       


def main(args=None):
    rclpy.init(args=args)

    aruco_detector = ArUcoDetector()

    rclpy.spin(aruco_detector)

    aruco_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()