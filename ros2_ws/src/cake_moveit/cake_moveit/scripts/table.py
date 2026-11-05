"""The table the arm is mounted on, in Gazebo and in MoveIt's planning scene.

This file is complete. Copy it into your package with the other starter
files; you do not need to edit it or run it yourself:

    cp scaffolds/table.py ros2_ws/src/lab05_moveit/lab05_moveit/scripts/

motion_planner.py and m5_detour.py call add_table() once MoveIt is ready. It
does two separate things, because Gazebo and MoveIt keep separate models of the
world (see the manual's Section 1.2):

  1. Gazebo: spawns a static table whose top is level with the base of the
     arm, so the simulated robot has something to stand on. Skipped if the
     table is already there.
  2. MoveIt: adds a matching collision box, so the planner will not plan a
     motion that puts the arm through the tabletop.

The table is the same size and height as the one Lab 6 uses.
"""

from __future__ import annotations

import shutil
import subprocess

from pymoveit2 import MoveIt2
from rclpy.node import Node

TABLE_SIZE = (1.0, 1.0, 0.05)  # x, y, thickness in meters
ROBOT_BASE_HEIGHT = 0.30  # the simulation spawns the arm 0.30 m above the ground
GAZEBO_WORLD = "empty"
GAZEBO_MODEL = "lab05_table"
MOVEIT_OBJECT = "table"

# Single-quoted attributes, because this whole string is quoted with double
# quotes inside the protobuf request below.
_TABLE_SDF = (
    "<?xml version='1.0'?>"
    "<sdf version='1.8'><model name='{name}'><static>true</static>"
    "<link name='top'>"
    "<collision name='c'><geometry><box><size>{x} {y} {z}</size></box></geometry></collision>"
    "<visual name='v'><geometry><box><size>{x} {y} {z}</size></box></geometry>"
    "<material><ambient>0.55 0.45 0.35 1</ambient><diffuse>0.55 0.45 0.35 1</diffuse></material>"
    "</visual></link></model></sdf>"
)


def _gazebo_has_table() -> bool:
    result = subprocess.run(
        ["gz", "model", "--list"], capture_output=True, text=True, check=False
    )
    return GAZEBO_MODEL in result.stdout


def _spawn_in_gazebo(node: Node) -> None:
    if shutil.which("gz") is None:
        node.get_logger().warn("The gz command is not available, so no table was added to Gazebo.")
        return
    if _gazebo_has_table():
        node.get_logger().info("Gazebo already has the table.")
        return

    sx, sy, sz = TABLE_SIZE
    sdf = _TABLE_SDF.format(name=GAZEBO_MODEL, x=sx, y=sy, z=sz)
    center_z = ROBOT_BASE_HEIGHT - sz / 2.0
    request = (
        f'sdf: "{sdf}" name: "{GAZEBO_MODEL}" allow_renaming: false '
        f"pose {{ position {{ x: 0.0 y: 0.0 z: {center_z} }} }}"
    )
    result = subprocess.run(
        [
            "gz", "service", "-s", f"/world/{GAZEBO_WORLD}/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", request,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if "data: true" in result.stdout:
        node.get_logger().info("Added the table to Gazebo.")
    else:
        node.get_logger().warn(
            "Could not add the table to Gazebo, so the arm has nothing to stand on "
            f"there. MoveIt still avoids it. gz said: {result.stdout.strip()} "
            f"{result.stderr.strip()}"
        )


def add_table(node: Node, moveit2: MoveIt2) -> None:
    """Put the table in Gazebo and in MoveIt's planning scene."""
    _spawn_in_gazebo(node)
    sx, sy, sz = TABLE_SIZE
    # The tabletop sits 0.1 mm below base_link so that the robot's own base is
    # not reported as touching it.
    moveit2.add_collision_box(
        id=MOVEIT_OBJECT,
        size=(sx, sy, sz),
        position=(0.0, 0.0, -0.0001 - sz / 2.0),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
    )
    node.get_logger().info("Added the table to MoveIt's planning scene.")
