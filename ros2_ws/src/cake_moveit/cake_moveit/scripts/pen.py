"""A pen on the end effector: draws the path the gripper actually travels in RViz.

This file is complete. Copy it into your package and run it in its own terminal;
you do not need to edit it.

    cp scaffolds/pen.py ros2_ws/src/lab05_moveit/lab05_moveit/scripts/
    ros2 run lab05_moveit pen

While the pen is down, the node looks up where end_effector_link is relative to
base_link about 20 times a second and extends a line through those points. It
records the motion the controllers produced, not the motion you asked for, so a
straight Cartesian segment draws a straight line and a joint-space move draws the
curve the gripper really swept.

Control it from a terminal:

    ros2 service call /pen/down std_srvs/srv/Trigger
    ros2 service call /pen/up std_srvs/srv/Trigger
    ros2 service call /pen/clear std_srvs/srv/Trigger
    ros2 param set /pen color "[0.9, 0.1, 0.1]"   # takes effect at the next pen down

or from your own node with PenClient (see motion_planner.py):

    self.pen = PenClient(self)
    self.pen.set_color(0.9, 0.1, 0.1)   # optional; applies to the next down()
    self.pen.down()

The repository's lab05.rviz already displays /pen_trail. Load it in RViz
with File > Open Config.
"""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Point
from rcl_interfaces.srv import SetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


class Pen(Node):
    def __init__(self) -> None:
        super().__init__("pen")
        self.declare_parameter("frame", "base_link")
        self.declare_parameter("tip", "end_effector_link")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("min_step", 0.002)  # meters between recorded points
        self.declare_parameter("width", 0.010)  # line width, meters
        self.declare_parameter("color", [0.0, 0.33, 0.61])  # red, green, blue in 0-1

        self._frame = self.get_parameter("frame").value
        self._tip = self.get_parameter("tip").value
        self._min_step = float(self.get_parameter("min_step").value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_publisher(Marker, "pen_trail", 10)

        self._strokes: list[Marker] = []
        self._down = False

        self.create_service(Trigger, "pen/down", self._on_down)
        self.create_service(Trigger, "pen/up", self._on_up)
        self.create_service(Trigger, "pen/clear", self._on_clear)
        rate = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / rate, self._sample)
        # RViz only shows markers published after it subscribes, so resend
        # everything once a second in case RViz was opened late.
        self.create_timer(1.0, self._republish_all)

        self.get_logger().info(
            f"Pen ready: tracing {self._tip} in {self._frame}. Pen is up; "
            "call /pen/down to start drawing."
        )

    def _tip_position(self) -> Point | None:
        try:
            transform = self._tf_buffer.lookup_transform(self._frame, self._tip, Time())
        except TransformException as exc:
            self.get_logger().warn(
                f"Cannot find {self._tip} in {self._frame} yet ({exc}). "
                "Is the simulation running?",
                throttle_duration_sec=5.0,
            )
            return None
        t = transform.transform.translation
        return Point(x=t.x, y=t.y, z=t.z)

    def _new_stroke(self) -> Marker:
        r, g, b = (float(c) for c in self.get_parameter("color").value)
        marker = Marker()
        marker.header.frame_id = self._frame
        marker.ns = "pen"
        marker.id = len(self._strokes)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(self.get_parameter("width").value)
        marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        return marker

    def _on_down(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._down:
            self._strokes.append(self._new_stroke())
            self._down = True
        response.success = True
        response.message = "pen down"
        return response

    def _on_up(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._down = False
        response.success = True
        response.message = "pen up"
        return response

    def _on_clear(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        delete_all = Marker()
        delete_all.header.frame_id = self._frame
        delete_all.ns = "pen"
        delete_all.action = Marker.DELETEALL
        self._publisher.publish(delete_all)
        self._strokes.clear()
        if self._down:
            self._strokes.append(self._new_stroke())
        response.success = True
        response.message = "pen trail cleared"
        return response

    def _sample(self) -> None:
        if not self._down:
            return
        point = self._tip_position()
        if point is None:
            return
        stroke = self._strokes[-1]
        if stroke.points:
            last = stroke.points[-1]
            step = ((point.x - last.x) ** 2 + (point.y - last.y) ** 2
                    + (point.z - last.z) ** 2) ** 0.5
            if step < self._min_step:
                return
        stroke.points.append(point)
        # A line strip needs two points before RViz draws it.
        if len(stroke.points) >= 2:
            self._publisher.publish(stroke)

    def _republish_all(self) -> None:
        for stroke in self._strokes:
            if len(stroke.points) >= 2:
                self._publisher.publish(stroke)


class PenClient:
    """Calls the pen services from inside another node.

    Your node's executor must already be spinning in a background thread, as it
    is in motion_planner.py, because each call waits for the pen's reply.
    """

    def __init__(self, node: Node) -> None:
        self._node = node
        self._down = False
        self._clients = {
            name: node.create_client(Trigger, f"/pen/{name}")
            for name in ("down", "up", "clear")
        }
        self._param_client = node.create_client(SetParameters, "/pen/set_parameters")

    def _call(self, name: str) -> bool:
        client = self._clients[name]
        if not client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().warn(
                f"/pen/{name} is not available. Start the pen in another terminal "
                "with: ros2 run lab05_moveit pen"
            )
            return False
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.done() and future.result().success

    def down(self) -> bool:
        self._down = self._call("down")
        return self._down

    def up(self) -> bool:
        # Nothing to lift if the pen was never put down, and saying so twice
        # when the pen node is not running helps nobody.
        if not self._down:
            return True
        self._down = False
        return self._call("up")

    def clear(self) -> bool:
        return self._call("clear")

    def set_color(self, red: float, green: float, blue: float) -> bool:
        """Set the color of the next line, each value from 0 to 1. The change
        takes effect at the next down()."""
        client = self._param_client
        if not client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().warn(
                "The pen is not running, so its color cannot be set. Start it with: "
                "ros2 run lab05_moveit pen"
            )
            return False
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                "color", Parameter.Type.DOUBLE_ARRAY, [float(red), float(green), float(blue)]
            ).to_parameter_msg()
        ]
        future = client.call_async(request)
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.done() and all(result.successful for result in future.result().results)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Pen()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
