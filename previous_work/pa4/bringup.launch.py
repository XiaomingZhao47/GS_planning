#!/usr/bin/env python3
"""
PA4 self-contained bringup — Stage + wall_follower + occupancy_grid_mapper + RViz.
All referenced files live in this folder.

    ros2 launch /home/xiaoming/ros2_ws/src/pa4/pa4_bringup/bringup.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def generate_launch_description():
    world = LaunchConfiguration("world")
    rviz_cfg = LaunchConfiguration("rviz")
    follower_cfg = LaunchConfiguration("follower")

    args = [
        DeclareLaunchArgument(
            "world",
            default_value=os.path.join(THIS_DIR, "2017-02-11-00-31-57"),
            description="Stage world path WITHOUT .world extension"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("follower", default_value="true"),
    ]

    stage_pkg = get_package_share_directory("stage_ros2")
    stage_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stage_pkg, "launch", "stage.launch.py")),
        launch_arguments={
            "world": world,
            "enforce_prefixes": "false",
            "one_tf_tree": "false",
        }.items(),
    )

    wall_follower = ExecuteProcess(
        cmd=["python3", os.path.join(THIS_DIR, "wall_follower.py")],
        output="screen",
        condition=IfCondition(follower_cfg),
    )

    mapper = ExecuteProcess(
        cmd=["python3", os.path.join(THIS_DIR, "occupancy_grid_mapper.py")],
        output="screen",
    )

    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        output="screen", condition=IfCondition(rviz_cfg),
    )

    return LaunchDescription(args + [stage_launch, wall_follower, mapper, rviz])
