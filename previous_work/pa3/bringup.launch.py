#!/usr/bin/env python3
"""
PA3 bringup — Stage + local map publisher + map→odom static TF 

Run as:
    ros2 launch /home/xiaoming/ros2_ws/src/pa3/bringup.launch.py

    rviz:=true|false      (default true)
    map_yaml:=<path>      (default <this_dir>/maze.yml)
    world:=<path-no-ext>  (default <this_dir>/maze)
    spawn_xy:=x,y         (default 2.0,2.0 — Stage's odom anchor)

The map is published by the local map_publisher.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def generate_launch_description():
    world = LaunchConfiguration("world")
    rviz_cfg = LaunchConfiguration("rviz")

    args = [
        DeclareLaunchArgument("map_yaml", default_value=os.path.join(THIS_DIR, "maze.yml")),
        DeclareLaunchArgument("world", default_value=os.path.join(THIS_DIR, "maze"),
                              description="Stage world path WITHOUT .world extension"),
        DeclareLaunchArgument("spawn_xy", default_value="2.0,2.0",
                              description="Robot spawn pose in maze.world — map→odom anchor"),
        DeclareLaunchArgument("rviz", default_value="true"),
    ]

    # stage simulator 
    stage_pkg = get_package_share_directory("stage_ros2")
    stage_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stage_pkg, "launch", "stage.launch.py")),
        launch_arguments={
            "world": world,
            "enforce_prefixes": "false",
            # one_tf_tree:=true would prefix frames with the robot's name
            "one_tf_tree": "false",
        }.items(),
    )

    # local map publisher 
    def _make_map_pub(context):
        yaml_path = context.launch_configurations["map_yaml"]
        return [ExecuteProcess(
            cmd=["python3", os.path.join(THIS_DIR, "map_publisher.py"),
                 "--ros-args", "-p", f"yaml_filename:={yaml_path}"],
            output="screen",
        )]
    map_pub = OpaqueFunction(function=_make_map_pub)

    # map to odom static transform
    # Stage publishes odom relative to the robot's spawn pose
    def _make_static_tf(context):
        x, y = (s.strip() for s in context.launch_configurations["spawn_xy"].split(","))
        return [Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom",
            arguments=[x, y, "0", "0", "0", "0", "map", "odom"],
            output="screen",
        )]
    map_to_odom = OpaqueFunction(function=_make_static_tf)

    # RViz 
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(rviz_cfg),
    )

    return LaunchDescription(args + [stage_launch, map_pub, map_to_odom, rviz])
