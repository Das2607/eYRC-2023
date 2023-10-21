from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    controller_node = Node(
        package="hb_task_1b",
        executable="controller",
    )

    service_node = Node(
        package="hb_task_1b",
        executable="service_node",
    )
    ld.add_action(service_node)
    ld.add_action(controller_node)

    return ld
