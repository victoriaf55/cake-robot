"""
Final project for ME442L Intro to Robotics. Programs Kinova Gen3 Lite arm to trace cookie cake design

Run with gazebo simulation and MoveIt already running: ros2 run cake_moveit draw_design
"""

import time

from cake_moveit.scripts.pen import PenClient

from cake_moveit.scripts.arm import (
    Y_MIN,
    Y_MAX,
    Z_MIN,
    Z_MAX, 
    PRESS_X,
    HOVER_X,
    RETRACT,
    WORK,
    ArmNode,
    init_ros,
    run,
)

NETID = "vyf"

class WriteInitials(ArmNode):
    def __init__(self) -> None:
        super().__init__("write_initials")
        self.pen = PenClient(self)
        self.get_logger().info(f"Midterm write initials started ({NETID}).")


    def start(self) -> bool:
        """From wherever the arm is, fold to RETRACT, then go to WORK.
        """
        return self.move_to_joints(RETRACT) and self.move_to_joints(WORK)

    def draw_stroke(self, 
                    points: list[tuple[float, float]],
                    pen_up_name: str | None=None,
                    pen_down_name: str | None=None) -> bool:
        """Draw one connected stroke through (y, z) board coordinates."""
        # Make sure two points are input
        if len(points) < 2:
            self.get_logger().error("A stroke needs at least two points.")
            return False

        # Check that points are within writing box bounds
        for y, z in points:
            if not Y_MIN <= y <= Y_MAX or not Z_MIN <= z <= Z_MAX:
                self.get_logger().error(
                    f"Point ({y:.3f}, {z:.3f}) is outside the writing box."
                )
                return False

        start_y, start_z = points[0]

        # Travel in front of the board
        if not self.move_to_pose(HOVER_X, start_y, start_z):
            return False
        # For recording poses
        if pen_up_name is not None:
            self.record_key_pose(pen_up_name)    
        # Press straight into board
        if not self.move_to_pose(PRESS_X, start_y, start_z, cartesian=True):
            return False
        # For recording poses
        if pen_down_name is not None:
            self.record_key_pose(pen_down_name)

        # Draw every connected segment as a full Cartesian path
        self.pen.down()
        current_y, current_z = start_y, start_z

        for y, z in points[1:]:
            if not self.move_to_pose(PRESS_X, y, z, cartesian=True):
                self.pen.up()
                self.move_to_pose(HOVER_X, current_y, current_z, cartesian=True)
                return False

        # Lift before traveling to a disconnected stroke
        self.pen.up()
        end_y, end_z = points[-1]
        return self.move_to_pose(HOVER_X, end_y, end_z, cartesian=True)


    def record_key_pose(self, name: str) -> None:
        """Log measured joint values for one required kinematics key pose."""
        time.sleep(0.25)
        joints = self.joint_positions()
        if joints is None:
            self.get_logger().error(f"Could not read joint values for {name}.")
            return

        values = ", ".join(f"{joint:.6f}" for joint in joints)
        self.get_logger().info(f"{name} = [{values}]")


    def run_initial_1(self) -> None:
        """Write V as one connected stroke on the left side of the board."""
        self.get_logger().info("Writing initial 1: V")
        if not self.start():
            return

        v_stroke = [
            (0.18, 0.26),  # top-left
            (0.10, 0.08),  # bottom point
            (0.02, 0.26),  # top-right
        ]

        self.pen.clear()
        # Draw "V". Record pen up and pen down position
        if not self.draw_stroke(v_stroke,
                                pen_up_name="pen_up_first_letter",
                                pen_down_name="pen_down_first_letter"
        ):
            self.get_logger().error("Could not finish V.")
            return
        self.get_logger().info("Finished V.")

    def run_initial_2(self) -> None:
        """Write F using three disconnected strokes."""
        self.get_logger().info("Writing initial 2: F")
        # if not self.start():
        #     return

        vertical = [
            (-0.04, 0.26),  # top-left
            (-0.04, 0.08),  # bottom-left
        ]
        top_bar = [
            (-0.04, 0.26),  # left
            (-0.19, 0.26),  # right
        ]
        middle_bar = [
            (-0.04, 0.17),  # left
            (-0.16, 0.17),  # right
        ]

        # Record pen down position
        if not self.draw_stroke(vertical, pen_down_name="pen_down_second_letter"):
            self.get_logger().error("Could not finish F.")
            return
        for stroke in (top_bar, middle_bar):
            if not self.draw_stroke(stroke):
                self.get_logger().error("Could not finish F.")
                return
        if not self.move_to_joints(RETRACT):
            self.get_logger().error("Could not retract after F.")
            return
        self.get_logger().info("Finished F.")


def main() -> None:
    init_ros()
    node = WriteInitials()

    run(
        node,
        {1: node.run_initial_1, 2: node.run_initial_2},
        default=[1, 2],
    )


if __name__ == "__main__":
    main()