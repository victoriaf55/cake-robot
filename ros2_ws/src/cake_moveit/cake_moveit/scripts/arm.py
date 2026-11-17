"""MoveIt motion helpers for the cake-decorating arm."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionFK
from pymoveit2 import MoveIt2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState


ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Tool local +Z points downward in base_link (quaternion x, y, z, w).
TOP_DOWN = (0.0, 1.0, 0.0, 0.0)
SPEED = 0.10


class ArmNode(Node):
    """A ROS 2 node with MoveIt and measured joint positions."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ARM_JOINTS,
            base_link_name="base_link",
            end_effector_name="end_effector_link",
            group_name="arm",
        )
        self._joint_positions: dict[str, float] = {}
        self._joint_stamp = 0.0
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self._fk_client = self.create_client(GetPositionFK, "compute_fk")

    # ------------------------------------------------------------ measured state

    def _on_joint_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self._joint_positions[name] = position
        # The robot's own clock: simulated time in Gazebo, real time on the arm.
        self._joint_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


    def joint_positions(self) -> list[float] | None:
        """Return measured arm joint positions in radians, ordered joint_1 through joint_6."""
        positions = [self._joint_positions.get(name) for name in ARM_JOINTS]
        if any(position is None for position in positions):
            return None
        return [float(position) for position in positions]


    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Wait for MoveIt to answer before sending it anything."""
        if not self._fk_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"MoveIt did not start within {timeout:.0f} s. Is the sim.launch.py "
                "MoveIt launch still running in its own terminal?"
            )
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.moveit2.compute_fk() is not None:
                self.get_logger().info("MoveIt is ready.")
                return True
            time.sleep(1.0)
        self.get_logger().error(
            "MoveIt answered but could not compute the arm's current pose. Check "
            "the MoveIt terminal for errors."
        )
        return False

    # ------------------------------------------------------------------- motion

    def move_to_joints(self, joint_positions: list[float], attempts: int = 3) -> bool:
        """Plan and execute a joint-space motion at the safe speed limit."""
        self.moveit2.max_velocity = SPEED
        self.moveit2.max_acceleration = SPEED
        try:
            for attempt in range(1, attempts + 1):
                trajectory = self.moveit2.plan(joint_positions=joint_positions)
                if trajectory is None:
                    self.get_logger().warn(
                        f"Planning attempt {attempt} of {attempts} failed."
                    )
                    continue
                self.moveit2.execute(trajectory)
                if self.moveit2.wait_until_executed():
                    return True
                self.get_logger().warn(
                    f"Execution attempt {attempt} of {attempts} failed."
                )
        finally:
            self.moveit2.max_velocity = 0.0
            self.moveit2.max_acceleration = 0.0

        self.get_logger().error("Could not plan or execute this joint-space motion.")
        return False

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        cartesian: bool = False,
        attempts: int = 3,
    ) -> bool:
        """Move end_effector_link to (x, y, z) in base_link, tool pointing down.

        cartesian=True moves in a straight line and fails rather than bend the
        path. Velocity and acceleration scaling use SPEED.
        """
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = TOP_DOWN
        self.moveit2.max_velocity = SPEED
        self.moveit2.max_acceleration = SPEED
        try:
            for attempt in range(1, attempts + 1):
                trajectory = self.moveit2.plan(
                    pose=pose,
                    cartesian=cartesian,
                    max_step=0.005,
                    cartesian_fraction_threshold=1.0 if cartesian else 0.0,
                )
                if trajectory is None:
                    self.get_logger().warn(
                        f"Planning attempt {attempt} of {attempts} to "
                        f"({x:.3f}, {y:.3f}, {z:.3f}) failed."
                    )
                    continue
                self.moveit2.execute(trajectory)
                if self.moveit2.wait_until_executed():
                    return True
                self.get_logger().warn(f"Execution attempt {attempt} of {attempts} failed.")
        finally:
            self.moveit2.max_velocity = 0.0
            self.moveit2.max_acceleration = 0.0
        kind = "straight-line" if cartesian else "pose-goal"
        self.get_logger().error(
            f"Could not plan a {kind} motion to ({x:.3f}, {y:.3f}, {z:.3f})."
        )
        return False



def run(
    node: ArmNode,
    milestones: dict[int, Callable[[], object]],
    default: list[int],
) -> None:
    """Spin the node, then run the milestones named on the command line, or
    the `default` ones when none are named."""
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    own_args = sys.argv[1:]
    if "--ros-args" in own_args:
        own_args = own_args[: own_args.index("--ros-args")]
    requested: list[int] = []
    for argument in own_args:
        if argument.isdigit() and int(argument) in milestones:
            requested.append(int(argument))
        else:
            valid = ", ".join(str(number) for number in sorted(milestones))
            node.get_logger().error(
                f"'{argument}' is not a milestone. Pass any of {valid}, or nothing to run all."
            )
            node.destroy_node()
            rclpy.try_shutdown()
            sys.exit(2)
    if not requested:
        requested = default

    try:
        if node.wait_until_ready():
            for number in requested:
                milestones[number]()
                time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def init_ros() -> None:
    """Start ROS with a Ctrl+C handler that exits at once.

    rclpy's own handler tears the context down under a motion that is still
    running, which leaves the node waiting for a result that can never arrive.
    """
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def stop(_signum: int, _frame: object) -> None:
        print("Interrupted. Stopping; the arm holds its position.", flush=True)
        os._exit(130)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
