"""Kinematics of the Gen3 Lite for my midterm.

The grader reads the values below without running this file, so write numbers
directly. You may use pi, + - * /, and np.array([...]). Lengths are in meters
and angles in radians. Replace every None.
"""
from math import pi

# One row per joint, joint 1 first, with the lecture's columns (theta, d, a, alpha).
# theta_i = q_i + theta_offset, where q_i is the joint value reported in
# /joint_states. Frame 0 is base_link. The last frame has its origin at
# end_effector_link and its z axis along the marker.
# For each intermediate frame (a fixed transform with no joint of its own), insert a row
# with "fixed": True at its position in the chain. Its theta is theta_offset alone.
DH = [
    {"theta_offset": None, "d": None, "a": None, "alpha": None},  # joint 1
    {"theta_offset": None, "d": None, "a": None, "alpha": None},  # joint 2
    {"theta_offset": None, "d": None, "a": None, "alpha": None},  # joint 3
    {"theta_offset": None, "d": None, "a": None, "alpha": None},  # joint 4
    {"theta_offset": None, "d": None, "a": None, "alpha": None},  # joint 5
    {"theta_offset": None, "d": None, "a": None, "alpha": None},  # joint 6
    # {"theta_offset": 0, "d": 0.1, "a": 0, "alpha": 0, "fixed": True},  # example
]

# Your three key poses: joint values q1 to q6 in radians, as in /joint_states.
# Rename the keys to describe each pose.
KEY_POSES = {
    "key_pose_1": [None, None, None, None, None, None],
    "key_pose_2": [None, None, None, None, None, None],
    "key_pose_3": [None, None, None, None, None, None],
}

# The Jacobian at one of your key poses, in base_link, for the origin of
# end_effector_link. Rows: vx, vy, vz, wx, wy, wz. Columns: joints 1 to 6.
JACOBIAN_POSE = "key_pose_1"
JACOBIAN = [
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
]
