import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math
from tf_transformations import euler_from_quaternion
from my_robot_interfaces.srv import NextGoal
from numpy import angle

PI = math.pi


class HBTask1BController(Node):
    def __init__(self):
        super().__init__("hb_task1b_controller")
        # Initialising publisher and subscriber of cmd_vel and odom respectively
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(Odometry, "/odom", self.callback_odom, 10)

        # Declare a Twist message
        self.vel = Twist()

        # For maintaining control loop rate.
        self.rate = self.create_rate(1000)
        # Initialise variables that may be needed for the control loop
        self.timer = self.create_timer(0.01, self.go_to_goal)
        # For ex: x_d, y_d, theta_d (in **meters** and **radians**) for defining desired goal-pose.
        # and also Kp values for the P Controller

        # client for the "next_goal" service
        self.cli = self.create_client(NextGoal, "next_goal")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Service not available, waiting again...")

        self.req = NextGoal.Request()
        self.index = 0
        self.goal_arr = [0, 0, 0]
        self.kp = 2.0
        self.kp_r = 4.0
        self.x = 0
        self.y = 0
        self.z = 0
        self.w = 0
        self.yaw = 0
        self.v_x = 0
        self.v_y = 0
        self.v_w = 0
        self.enable_cont = True

    def send_request(self, index):
        self.req.request_goal = index
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def callback_odom(self, Odometry):
        """
        Callback func to record the yaw, x, y of the bot
        """
        w = Odometry.pose.pose.orientation.w
        x = Odometry.pose.pose.orientation.x
        y = Odometry.pose.pose.orientation.y
        z = Odometry.pose.pose.orientation.z
        self.x = Odometry.pose.pose.position.x
        self.y = Odometry.pose.pose.position.y
        self.z = Odometry.pose.pose.position.z
        self.yaw, __, _ = euler_from_quaternion([w, x, y, z])

    def difference_rotation(self, theta):
        """Calculates the error in pose for angular rotation in radians from (-pi, pi).

        Args:
            current_angle: The current angle in radians.
            desired_angle: The desired angle in radians.

        Returns:
            The error in angle in radians.
        """

        # Normalize the angles to the range (-pi, pi).
        current_angle = math.acos(math.cos(self.yaw))
        desired_angle = math.acos(math.cos(theta))

        error = desired_angle - current_angle
        if theta - self.yaw < 0:
            error *= -1

        return error

    def error_vx(self):
        return self.goal_arr[0] - self.x

    def error_vy(self):
        return self.goal_arr[1] - self.y

    def error_linear(self):
        return complex(self.error_vx(), self.error_vy())

    def vel_lin(self):
        vec_vel = self.error_linear()
        abs_vel = abs(vec_vel)

        # Unit velocity vector
        if abs_vel:
            scale_vel = vec_vel / abs_vel
        else:
            scale_vel = 0

        e_l = scale_vel * complex(math.cos(self.yaw), math.sin(self.yaw))
        # Pre vel checks
        mov_vel = self.check_vel(
            self.euclidean_distance(self.goal_arr[0], self.goal_arr[1]), 1.0
        )

        # Vel Hard Limits
        self.v_x = -self.check_vel(e_l.real * mov_vel, 5)
        self.v_y = -self.check_vel(e_l.imag * mov_vel, 5)

    def euclidean_distance(self, x, y):
        return math.sqrt(pow((x - self.x), 2) + pow((y - self.y), 2))

    def go_to_goal(self):
        if not self.enable_cont:
            return

        msg = Twist()
        self.vel_lin()
        msg.linear.x = self.kp * self.v_x
        msg.linear.y = self.kp * self.v_y
        msg.angular.z = self.kp_r * self.difference_rotation(float(self.goal_arr[2]))
        self.pub.publish(msg)

    def stop_move(self):
        self.enable_cont = False
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = 0.0
        self.pub.publish(msg)

    def start_move(self):
        self.enable_cont = True

    def check_vel(self, vel, max_vel=1.5):
        vel = vel * self.kp
        max_vel = float(max_vel)
        if vel > max_vel:
            return max_vel
        if vel < -max_vel:
            return -max_vel
        return vel

    def wait_for_rot(self, r_tol):
        rot = self.difference_rotation(self.goal_arr[2])
        if abs(rot) < r_tol:
            return True
        return False

    def is_at_goal(self, l_tol, r_tol):
        dist = self.euclidean_distance(self.goal_arr[0], self.goal_arr[1])
        if dist < l_tol and self.wait_for_rot(r_tol):
            return True
        return False


def main(args=None):
    rclpy.init(args=args)

    # Create an instance of the EbotController class
    ebot_controller = HBTask1BController()

    # Send an initial request with the index from ebot_controller.index
    ebot_controller.send_request(ebot_controller.index)

    ebot_controller.goal_arr = [0, 0, 0]

    while not ebot_controller.is_at_goal(0.1, 0.1):
        rclpy.spin_once(ebot_controller)

    ebot_controller.stop_move()

    # Main loop
    while rclpy.ok():
        # Check if the service call is done
        if ebot_controller.future.done():
            try:
                # response from the service call
                response = ebot_controller.future.result()
            except Exception as e:
                ebot_controller.get_logger().infselfo("Service call failed %r" % (e,))
            else:
                #########           GOAL POSE             #########
                x_goal = response.x_goal
                y_goal = response.y_goal
                theta_goal = response.theta_goal
                ebot_controller.flag = response.end_of_list
                ####################################################

                ebot_controller.goal_arr = [x_goal, y_goal, theta_goal]

                ebot_controller.start_move()
                while not ebot_controller.is_at_goal(0.1, 0.1):
                    rclpy.spin_once(ebot_controller)
                ebot_controller.stop_move()

                # Sleep for 1 sec
                for i in range(15):
                    time.sleep(0.1)
                    ebot_controller.get_logger().info(
                        f"STAB {(i*100)//14}% {ebot_controller.flag}"
                    )

                ############     DO NOT MODIFY THIS       #########
                ebot_controller.index += 1
                if ebot_controller.flag == 1:
                    ebot_controller.index = 0
                ebot_controller.send_request(ebot_controller.index)
                ####################################################

        # Spin once to process callbacks
        rclpy.spin_once(ebot_controller)

    # Destroy the node and shut down ROS
    ebot_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
