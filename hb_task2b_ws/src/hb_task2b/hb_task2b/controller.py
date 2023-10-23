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
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		controller.py
# Functions:
#			[ Comma separated list of functions in this file ]
# Nodes:		Add your publishing and subscribing node


################### IMPORT MODULES #######################

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Wrench
import time
import math
from tf_transformations import euler_from_quaternion
from my_robot_interfaces.msg import Goal  
import numpy as np           

# You can add more if required
##############################################################


# Initialize Global variables


################# ADD UTILITY FUNCTIONS HERE #################

##############################################################


# Define the HBController class, which is a ROS node
class HBController(Node):
    def __init__(self, name="temp"):
        super().__init__(f'hb_controller_{name}')
        
        # Initialze Publisher and Subscriber
        # NOTE: You are strictly NOT-ALLOWED to use "cmd_vel" or "odom" topics in this task
	    #	Use the below given topics to generate motion for the robot.
	    #   /hb_bot_1/left_wheel_force,
	    #   /hb_bot_1/right_wheel_force,
	    #   /hb_bot_1/rear_wheel_force
        self.left_wheel = self.create_publisher(Wrench, f"/hb_bot_{name}/left_wheel_force", 10)
        self.right_wheel = self.create_publisher(Wrench, f"/hb_bot_{name}/right_wheel_force", 10)
        self.rear_wheel = self.create_publisher(Wrench, f"/hb_bot_{name}/rear_wheel_force", 10)

        self.sub = self.create_subscription(Pose, f"/detected_aruco_{name}", self.pose_callback, 10)
        self.goal_sub = self.create_subscription(Goal, f"/hb_bot_{name}/goal", self.goal_callback, 10)


        self.iter = 0

        # For maintaining control loop rate.
        self.rate = self.create_rate(1000)
        self.timer = self.create_timer(0.02, self.move)


        # client for the "next_goal" service
        # self.cli = self.create_client(NextGoal, 'next_goal')      
        # self.req = NextGoal.Request() 
        self.index = 0

        # Goal array [x, y, w]
        self.goal_array = [0, 0, 0]
        self.goal_arrays_x = [0]
        self.goal_arrays_y = [0]
        self.goal_arrays_w = 0

        # Enable Controller
        self.controller_enable = False
        self.got_goal = False

        # Current Pose
        self.pose = [0, 0, 0]
        self.pose_detected = False

        # Distance from centroid to wheel, (Measured in blender)
        self.wheel_dist = 0.6

        # Conversion Matrix
        self.conv_mtx = np.array([
            [-self.wheel_dist, 1.0, 0],
            [-self.wheel_dist, -0.5, -math.sin(math.pi/3.0)],
            [-self.wheel_dist, -0.5, math.sin(math.pi/3.0)]
        ])

    def pose_callback(self, pose):
        self.pose = [pose.position.x, pose.position.y, pose.orientation.z]
        self.pose_detected = True
    
    def start_move(self):
        # self.get_logger().info("MOVING")
        self.controller_enable = True and self.pose_detected and self.got_goal
        # print(self.controller_enable)
    
    def goal_callback(self, goal):
        self.goal_arrays_x = goal.x
        self.goal_arrays_y = goal.y
        self.goal_arrays_w = goal.theta

        self.got_goal = True
    # Method to create a request to the "next_goal" service
    # def send_request(self, request_goal):
    #     self.req.request_goal = request_goal
    #     self.future = self.cli.call_async(self.req)

        
    def move(self):
        self.iter += 1
        if (self.controller_enable == False):
            if (self.iter % 1 == 0):
                self.get_logger().info(f"controller: {self.controller_enable} pose: {self.pose_detected} goal:{self.got_goal}")
                self.get_logger().info(f"{self.goal_array}")
            self.stop_move()
            return
        msg = self.inverse_kinematics()
        self.left_wheel.publish(msg[0])
        self.right_wheel.publish(msg[1])
        self.rear_wheel.publish(msg[2])

    def error_vx(self):
        return float(self.goal_array[0] - self.pose[0])
    
    def error_vy(self):
        return float(self.goal_array[1] - self.pose[1])
    
    def error_wz(self):
        current_angle = math.acos(math.cos(self.pose[2]))
        desired_angle = math.acos(math.cos(self.goal_array[2]))

        error = desired_angle - current_angle
        if self.goal_array[2] - self.pose[2] < 0:
            error *= -1

        return error * 1

    def stop_move(self):
        msg = Wrench()
        self.left_wheel.publish(msg)
        self.right_wheel.publish(msg)
        self.rear_wheel.publish(msg)
        self.controller_enable = False

    def is_at_goal(self, tol, r_tol):
        ev_x = self.error_vx()
        ev_y = self.error_vy()
        ew_z = self.error_wz()

        is_true_if = (abs(ev_x) < tol) and (abs(ev_y) < tol) and (abs(ew_z) < r_tol)

        return is_true_if


    def inverse_kinematics(self, kp = 50):
        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 
        #	-> Use the target velocity you calculated for the robot in previous task, and
        #	Process it further to find what proportions of that effort should be given to 3 individuals wheels !!
        #	Publish the calculated efforts to actuate robot by applying force vectors on provided topics
        ############################################
        
        # Rotation Matrix to transform area coords to bot coords
        rot_matrix = np.array([
            [1,0,0],
            [0, math.cos(self.pose[2]), math.sin(self.pose[2])],
            [0, -math.sin(self.pose[2]), math.cos(self.pose[2])]
        ])

        # Goal Matrix 
        goal_matrix = np.array([
            [math.pow(self.error_wz(), 1) * kp],
            [math.pow(self.error_vx(), 1) * kp],
            [math.pow(self.error_vy(), 1) * kp]
        ])

        # Final Transform
        final_matrix = self.conv_mtx @ rot_matrix @ goal_matrix

        final_matrix = final_matrix  

        final_matrix /= self.force_limiter(final_matrix)
        # Convert Matrix to Wrench

        # wheel in order of [b, l, r]

        msg_l = Wrench()
        msg_r = Wrench()
        msg_b = Wrench()
        msg_l.force.y = final_matrix.flatten()[1]
        msg_r.force.y = final_matrix.flatten()[2]
        msg_b.force.y = final_matrix.flatten()[0]
        # print(msg_l.force.y, msg_r.force.y, msg_b.force.y)
        # print(self.pose)
        return (msg_l, msg_r, msg_b)

    def force_limiter(self, final_matrix, MAX_F = 50.0):
        lf = abs(final_matrix.flatten()[1])
        rf = abs(final_matrix.flatten()[2])
        bf = abs(final_matrix.flatten()[0])

        tf = max(lf, rf, bf)

        if (tf <= 25.0):
            return 1
        # self.get_logger().warn(">>LIMITER<<")
        return (tf / MAX_F)


def main(args=None):
    rclpy.init(args=args)
    
    # Create an instance of the HBController class
    hb_controller_1 = HBController("1")
    hb_controller_2 = HBController("2")
    hb_controller_3 = HBController("3")

    bot_arr = [hb_controller_1, hb_controller_2, hb_controller_3]
   
    # Send an initial request with the index from hb_controller_1.index
    # hb_controller_1.send_request(hb_controller_1.index)
    # hb_controller_2.send_request(hb_controller_2.index)
    # hb_controller_3.send_request(hb_controller_2.index)
    
    # Main loop
    while rclpy.ok():

        # Check if the service call is done
        for i in range(len(bot_arr)):
            # if bot_arr[i].future.done():
            #     try:
            #         # response from the service call
            #         response = bot_arr[i].future.result()
            #     except Exception as e:
            #         hb_controller_1.get_logger().infselfo(
            #             'Service call failed %r' % (e,))
            #     else:
                    # #########           GOAL POSE             #########
            x_goal      = bot_arr[i].goal_arrays_x[bot_arr[i].index]
            y_goal      = bot_arr[i].goal_arrays_y[bot_arr[i].index]
            theta_goal  = bot_arr[i].goal_arrays_w
                    # ####################################################
            # time.sleep(0.01)

            bot_arr[i].goal_array = [x_goal/25.0 - 9, y_goal/25.0 - 9 , theta_goal]
            # print(bot_arr[i].goal_array)
            bot_arr[i].start_move()
            # print(bot_arr[i].goal_array)
            # print(x_goal)
                    # bot_arr[i].start_move()
                    # while not bot_arr[i].is_at_goal(0.2, 0.2):
                    #     rclpy.spin_once(bot_arr[i])
                    # bot_arr[i].stop_move()

            if bot_arr[i].is_at_goal(0.25, 0.25):
                # bot_arr[i].stop_move()
                bot_arr[i].index += 1
                if bot_arr[i].index == len(bot_arr[i].goal_arrays_x) :
                    bot_arr[i].index = 0
                    # bot_arr[i].send_request(bot_arr[i].index)
                    ####################################################

            # Spin once to process callbacks
            rclpy.spin_once(bot_arr[i])
    
    # Destroy the node and shut down ROS
    hb_controller_1.destroy_node()
    hb_controller_2.destroy_node()
    hb_controller_3.destroy_node()
    rclpy.shutdown()

# Entry point of the script
if __name__ == '__main__':
    main()


"""

        if bot_arr[i].future.done():
            try:
                # response from the service call
                response = bot_arr[i].future.result()
            except Exception as e:
                hb_controller_1.get_logger().infselfo(
                    'Service call failed %r' % (e,))
            else:
                #########           GOAL POSE             #########
                x_goal      = response.x_goal
                y_goal      = response.y_goal
                theta_goal  = response.theta_goal
                bot_arr[i].flag = response.end_of_list
                ####################################################
                
                bot_arr[i].goal_array = [x_goal/10, y_goal/10, theta_goal]

                bot_arr[i].start_move()
                while not bot_arr[i].is_at_goal(0.2, 0.2):
                    rclpy.spin_once(bot_arr[i])
                bot_arr[i].stop_move()

                ############     DO NOT MODIFY THIS       #########
                bot_arr[i].index += 1
                if bot_arr[i].flag == 1 :
                    bot_arr[i].index = 0
                bot_arr[i].send_request(bot_arr[i].index)
                ####################################################

"""