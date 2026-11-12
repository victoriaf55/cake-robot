"""Trace a basic horizontal circle using the configured cake geometry."""

import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from cake_moveit.scripts.arm import ArmNode, init_ros
from cake_moveit.scripts.cake_coordinates import CakeGeometry
from cake_moveit.scripts.pen import PenClient


class DrawCircle(ArmNode):
    """Move home, approach, draw a circle, lift, and return home."""

    def __init__(self):
        super().__init__('draw_circle')
        defaults = CakeGeometry()
        for name, value in vars(defaults).items():
            # A negative sentinel distinguishes an unmeasured tool offset.
            self.declare_parameter(name, -1.0 if value is None else value)
        self.declare_parameter('visualize_pen', False)
        self.pen = None

    def draw_circle(self):
        """Stop the sequence immediately if any planned motion fails."""
        values = {name: self.get_parameter(name).value
                  for name in vars(CakeGeometry())}
        try:
            geometry = CakeGeometry(**values)
        except ValueError as exc:
            self.get_logger().error(
                f'Invalid cake geometry: {exc}. Set nozzle_offset to the '
                'measured tool length in meters before moving.')
            return False
        targets = geometry.waypoints()['end_effector_waypoints']
        if targets is None:
            return False
        if self.get_parameter('visualize_pen').value:
            self.pen = PenClient(self)
            if not self.pen.clear():
                return False

        # Match the collision tabletop to the geometry used for target heights.
        self.moveit2.add_collision_box(
            id='table', size=(1.0, 1.0, 0.05),
            position=(0.0, 0.0, geometry.tabletop_z - 0.025),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id='base_link')

        if not self.move_to_pose(*targets['home']):
            return False
        if not self.move_to_pose(*targets['approach'], cartesian=True):
            return False
        if not self.move_to_pose(*targets['circle'][0], cartesian=True):
            return False
        if self.pen is not None and not self.pen.down():
            return False
        try:
            for point in targets['circle'][1:]:
                if not self.move_to_pose(*point, cartesian=True):
                    # Actual position may be partway through a failed segment;
                    # do not issue a recovery motion from an assumed position.
                    return False
        finally:
            if self.pen is not None:
                self.pen.up()
        if not self.move_to_pose(*targets['lift'], cartesian=True):
            return False
        if not self.move_to_pose(*targets['return_home'], cartesian=True):
            return False
        self.get_logger().info('Circle complete; nozzle is at home hover.')
        return True


def main():
    """Run one circle after MoveIt is ready."""
    init_ros()
    node = DrawCircle()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    success = False
    try:
        if node.wait_until_ready():
            success = node.draw_circle()
            if not success:
                node.get_logger().error('Circle sequence stopped before completion.')
    finally:
        executor.shutdown()
        thread.join()
        node.destroy_node()
        rclpy.try_shutdown()
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
