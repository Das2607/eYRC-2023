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
# Author List:	[ Shreyas Das ]
# Filename:		controller.py
# Functions:
#			[ Comma separated list of functions in this file ]
# Nodes:		Add your publishing and subscribing node


################### IMPORT MODULES #######################

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Wrench
import math
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
        
        self.id = name

        self.left_wheel = self.create_publisher(Wrench, f"/hb_bot_{name}/left_wheel_force", 10)
        self.right_wheel = self.create_publisher(Wrench, f"/hb_bot_{name}/right_wheel_force", 10)
        self.rear_wheel = self.create_publisher(Wrench, f"/hb_bot_{name}/rear_wheel_force", 10)

        self.sub = self.create_subscription(Pose, f"/detected_aruco_{name}", self.pose_callback, 10)
        self.goal_sub = self.create_subscription(Goal, f"/hb_bot_{name}/goal", self.goal_callback, 10)

        self.iter = 0

        # For maintaining control loop rate.
        self.rate = self.create_rate(1000)
        self.timer = self.create_timer(0.01, self.move)

        # Index for getting goal
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

        # Conversion Matrix to transform Bot coords to wheel velocity (wheel velocity is NOT angular velocity)
        self.conv_mtx = np.array([
            [-self.wheel_dist, 1.0, 0],
            [-self.wheel_dist, -0.5, -math.sin(math.pi/3.0)],
            [-self.wheel_dist, -0.5, math.sin(math.pi/3.0)]
        ])

    def pose_callback(self, pose):
        self.pose = [pose.position.x, pose.position.y, pose.orientation.z]
        self.pose_detected = True
    
    # Enable controller if pose is detected
    def start_move(self):
        self.controller_enable = True and self.pose_detected
    
    def goal_callback(self, goal):
        self.goal_arrays_x = goal.x
        self.goal_arrays_y = goal.y
        self.goal_arrays_w = goal.theta

        self.got_goal = True

    # Check if pose, goal is present and the controller is enabled
    def move(self):
        self.iter += 1
        if (self.controller_enable == False and self.got_goal == False):
            if (self.iter % 1 == 0):
                self.get_logger().info(f"controller: {self.controller_enable} pose: {self.pose_detected} goal:{self.got_goal}")
                self.get_logger().info(f"{self.goal_array}")
                self.iter = 0
            self.stop_move()
            return
        msg = self.inverse_kinematics()
        self.left_wheel.publish(msg[0])
        self.right_wheel.publish(msg[1])
        self.rear_wheel.publish(msg[2])


    # Calc error
    def error_vx(self):
        return float(self.goal_array[0] - self.pose[0])
    
    def error_vy(self):
        return float(self.goal_array[1] - self.pose[1])
    

    # OLD wz error func
    # def error_wz(self):
    #     current_angle = math.acos(math.cos(self.pose[2]))
    #     desired_angle = math.acos(math.cos(self.goal_array[2]))

    #     error = desired_angle - current_angle
    #     if self.goal_array[2] - self.pose[2] < 0:
    #         error *= -1

    #     return error * 2
    
    # New wz error func
    def error_wz(self):
        current_angle = math.sin(self.pose[2])
        desired_angle = math.sin(self.goal_array[2])
        error = desired_angle - current_angle
        error *= 3.0
        return error

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


    def inverse_kinematics(self, kp = 25):

        # Rotation Matrix to transform arena coords to bot coords
        rot_matrix = np.array([
            [1,0,0],
            [0, math.cos(self.pose[2]), math.sin(self.pose[2])],
            [0, -math.sin(self.pose[2]), math.cos(self.pose[2])]
        ])

        # Calc Kp controller values
        wz = self.error_wz() * kp
        vx = self.error_vx() * kp
        vy = self.error_vy() * kp

        # Goal Matrix defined in arena coords
        goal_matrix = np.array([
            [wz],
            [vx],
            [vy]
        ])

        # Final Transform: wheel coords <- bot coords <- arena coords
        final_matrix = self.conv_mtx @ rot_matrix @ goal_matrix

        final_matrix /= self.force_limiter(final_matrix)

        final_matrix *= self.force_leveler(final_matrix)

        # wheel in order of [b, l, r]
        msg_l = Wrench()
        msg_r = Wrench()
        msg_b = Wrench()
        msg_l.force.y = final_matrix.flatten()[1]
        msg_r.force.y = final_matrix.flatten()[2]
        msg_b.force.y = final_matrix.flatten()[0]
        return (msg_l, msg_r, msg_b)

    def force_limiter(self, final_matrix, MAX_F = 50.0):
        lf = abs(final_matrix.flatten()[1])
        rf = abs(final_matrix.flatten()[2])
        bf = abs(final_matrix.flatten()[0])

        tf = max(lf, rf, bf)

        if (tf <= 25.0):
            return 1
        return (tf / MAX_F)
    
    def force_leveler(self, final_matrix, desired_force = 45.0):
        lf = abs(final_matrix.flatten()[1])
        rf = abs(final_matrix.flatten()[2])
        bf = abs(final_matrix.flatten()[0])

        tf = math.sqrt(pow(lf, 2) + pow(rf, 2) + pow(bf, 2))

        return (desired_force / tf)


def main(args=None):
    rclpy.init(args=args)
    
    # Create an instance of the HBController class
    hb_controller_1 = HBController("1")
    hb_controller_2 = HBController("2")
    hb_controller_3 = HBController("3")

    bot_arr = [hb_controller_1, hb_controller_2, hb_controller_3]
    bot_is_done = [0, 0, 0]
    flag = True

    for i_ in range(len(bot_arr)):
        bot_arr[i_].start_move()
    
    # Main loop
    while rclpy.ok() and flag:

        # Scan and control all the bots
        for i in range(len(bot_arr)):
            # #########           GOAL POSE             #########
            x_goal      = bot_arr[i].goal_arrays_x[bot_arr[i].index]
            y_goal      = bot_arr[i].goal_arrays_y[bot_arr[i].index]
            theta_goal  = bot_arr[i].goal_arrays_w
            # ####################################################
            
            bot_arr[i].goal_array = [x_goal/30.0 - 8, y_goal/30.0 - 8, theta_goal]

            ####################################################

            if bot_arr[i].is_at_goal(0.3, 0.1):
                bot_arr[i].index += 1
                if bot_arr[i].index == len(bot_arr[i].goal_arrays_x) :
                    bot_is_done[i] = 1
                    bot_arr[i].index = 0

                    # Once all the bot has looped, end the while loop
                    # for k in bot_is_done:
                    #     if k == 0:
                    #         break
                    # else:
                    #     flag = False                   

            ####################################################

            # if bot_is_done[i] == 1:
            #     bot_arr[i].stop_move()
                # print(f"Done -> {i}")

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