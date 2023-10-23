from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    controller_node = Node(
        package="hb_task2a",
        executable="controller",
    )
    feedback = Node(
        package="hb_task2a",
        executable="feedback",
    )
    # service_node = Node(
    #     package="hb_task2a",
    #     executable="service",
    # )
    ld.add_action(controller_node)
    ld.add_action(feedback)
    # ld.add_action(service_node)

    return ld
