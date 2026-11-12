"""Generate provisional cake waypoints in base_link; never command the arm."""

import argparse
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class CakeGeometry:
    """Dimensions in meters; heights and center are relative to base_link."""

    center_x: float = 0.30
    center_y: float = 0.0
    tabletop_z: float = -0.01
    cake_height: float = 0.02
    diameter: float = 12 * 0.0254
    inset: float = 0.01
    nozzle_gap: float = 0.002
    clearance: float = 0.05
    segments: int = 180
    nozzle_offset: float | None = None

    def __post_init__(self):
        """Reject invalid geometry before generating targets."""
        for name, value in vars(self).items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.diameter <= 0 or self.cake_height <= 0:
            raise ValueError('diameter and cake_height must be positive')
        if not 0 <= self.inset < self.diameter / 2:
            raise ValueError('inset must be between zero and the cake radius')
        if not 0 <= self.nozzle_gap < self.clearance:
            raise ValueError('clearance must exceed the nonnegative nozzle gap')
        if not isinstance(self.segments, int) or self.segments < 3:
            raise ValueError('segments must be an integer of at least three')
        if self.nozzle_offset is not None and self.nozzle_offset <= 0:
            raise ValueError('nozzle_offset must be positive when supplied')

    def waypoints(self):
        """Return home, approach, closed XY circle, lift, and return targets."""
        radius = self.diameter / 2 - self.inset
        top_z = self.tabletop_z + self.cake_height
        draw_z = top_z + self.nozzle_gap
        hover_z = top_z + self.clearance
        home = [self.center_x, self.center_y, hover_z]
        # Start on the side nearest the base; traverse counterclockwise from +Z.
        circle = [
            [self.center_x + radius * math.cos(math.pi + 2 * math.pi * i / self.segments),
             self.center_y + radius * math.sin(math.pi + 2 * math.pi * i / self.segments),
             draw_z]
            for i in range(self.segments)
        ]
        circle.append(circle[0].copy())
        approach = [circle[0][0], circle[0][1], hover_z]
        tip = dict(home=home, approach=approach, circle=circle,
                   lift=approach.copy(), return_home=home.copy())
        result = {
            'frame_id': 'base_link',
            'units': 'meters',
            'provisional': True,
            'cake_top_z': top_z,
            'draw_radius': radius,
            'orientation_xyzw': [0.0, 1.0, 0.0, 0.0],
            'nozzle_offset': self.nozzle_offset,
            'tip_waypoints': tip,
            'end_effector_waypoints': None,
        }
        if self.nozzle_offset is not None:
            # Assumes the nozzle extends along end_effector_link local +Z.
            # The downward orientation maps this axis to base_link -Z.
            def convert(point):
                return [point[0], point[1], point[2] + self.nozzle_offset]
            result['end_effector_waypoints'] = {
                name: [convert(point) for point in value] if name == 'circle'
                else convert(value)
                for name, value in tip.items()
            }
        return result


def main():
    """Print a coordinate preview as JSON without requiring ROS or MoveIt."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = CakeGeometry()
    for name, value in vars(defaults).items():
        parser.add_argument('--' + name.replace('_', '-'), default=value,
                            type=int if name == 'segments' else float)
    try:
        geometry = CakeGeometry(**vars(parser.parse_args()))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(geometry.waypoints(), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
