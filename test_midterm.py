#!/usr/bin/env python3
"""Automated repository checks for the Midterm Project: Whiteboard Writing.

Run with: pytest test_midterm.py -v

These checks inspect the README, ROS 2 package, Python syntax, and tracked build output,
and check that kinematics.py is complete and that its key poses follow the handout's
rules. They do not check whether your DH table or Jacobian is correct; course staff
do that after the deadline. Course staff grade the writeup and the code from the
Gradescope PDF, and the whiteboard run at the lab station.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

WORKSPACE_SRC = Path("ros2_ws/src")
README_STARTER_FINGERPRINTS = ("Update this README", "[Your Name]")
IGNORED_PARTS = {"build", "install", "log", ".git", "__pycache__"}
NOT_STUDENT_CODE = {"setup.py"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path for path in root.rglob("*")
        if path.is_file() and not (set(path.relative_to(root).parts) & IGNORED_PARTS)
    ]


def find_packages(repo: Path) -> list[Path]:
    """Packages under ros2_ws/src, found by their package.xml."""
    src = repo / WORKSPACE_SRC
    packages = []
    for manifest in sorted(src.rglob("package.xml")) if src.is_dir() else []:
        package = manifest.parent
        if set(package.relative_to(repo).parts) & IGNORED_PARTS:
            continue
        packages.append(package)
    return packages


def _student_python(package: Path) -> list[Path]:
    """Non-empty Python files in a package, without setup.py and the test/ folder
    that ros2 pkg create generates."""
    return [
        path for path in _files_under(package)
        if path.suffix == ".py"
        and path.name not in NOT_STUDENT_CODE
        and "test" not in path.relative_to(package).parts[:-1]
        and _read_text(path).strip()
    ]


def check_package(repo: Path) -> list[str]:
    packages = find_packages(repo)
    if not packages:
        return [
            "Add your ROS 2 package under ros2_ws/src/, as in Lab 6: a folder with "
            "package.xml and your Python code (for example "
            "ros2_ws/src/midterm_writer/midterm_writer/write_initials.py)."
        ]

    errors: list[str] = []
    defines_function = False
    found_python = False
    for package in packages:
        for source in _student_python(package):
            found_python = True
            name = source.relative_to(repo).as_posix()
            try:
                tree = ast.parse(_read_text(source), filename=name)
            except SyntaxError as exc:
                errors.append(
                    f"{name} has a Python syntax error on line {exc.lineno}: {exc.msg}. "
                    "Fix it, then run the file again."
                )
                continue
            if any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(tree)
            ):
                defines_function = True

    if not found_python:
        names = ", ".join(package.relative_to(repo).as_posix() for package in packages)
        errors.append(
            f"No Python code found in {names}. Add the script that writes your initials "
            "to the package."
        )
    elif not defines_function and not errors:
        errors.append(
            "Your Python code defines no functions. The handout asks for at least one "
            "function; move a repeated piece of your program, such as drawing one "
            "stroke, into a function."
        )
    return errors


def check_required_files(repo: Path) -> list[str]:
    errors: list[str] = []

    readme = repo / "README.md"
    if not readme.is_file():
        errors.append("Add the missing required file: README.md")
    else:
        leftovers = [m for m in README_STARTER_FINGERPRINTS if m in _read_text(readme)]
        if leftovers:
            errors.append(
                "README.md still contains starter text: "
                + ", ".join(f"'{marker}'" for marker in leftovers)
                + ". Replace it with your name, NetID, and a 1-3 line summary."
            )

    errors.extend(check_package(repo))
    return errors


def check_repository_hygiene(repo: Path) -> list[str]:
    if not (repo / ".git").exists():
        return [
            "Run pytest from the root of your cloned repository. "
            "The grader could not find its .git directory."
        ]

    result = _run_git(repo, "ls-files")
    if result.returncode != 0:
        return [
            "The grader could not inspect tracked files with 'git ls-files'. "
            "Run 'git status' and resolve the reported Git problem first."
        ]

    forbidden_directories = {"build", "install", "log"}
    tracked = [Path(line) for line in result.stdout.splitlines() if line]
    prohibited = [
        path.as_posix()
        for path in tracked
        if ({part.casefold() for part in path.parts[:-1]} & forbidden_directories)
    ]
    if prohibited:
        shown = prohibited[:8]
        remainder = len(prohibited) - len(shown)
        summary = ", ".join(shown)
        if remainder:
            summary += f", and {remainder} more"
        return [
            "Remove generated build/, install/, and log/ files from Git tracking, "
            "then commit and push again. The files may remain on your VM if .gitignore "
            f"excludes them. Tracked generated files: {summary}"
        ]
    return []


# ------------------------------------------------------------------ kinematics.py

KINEMATICS_FILE = Path("kinematics.py")
RUN_COMMAND = "write_initials"

# The Gen3 Lite chain in the course image (kortex_description): the origin of each
# joint in its parent link as (xyz, rpy). Every joint turns about its own z axis.
GEN3_LITE_JOINTS = (
    ((0.0, 0.0, 0.12825), (0.0, 0.0, 0.0)),
    ((0.0, -0.03, 0.115), (1.5708, 0.0, 0.0)),
    ((0.0, 0.28, 0.0), (-3.1416, 0.0, 0.0)),
    ((0.0, -0.14, 0.02), (1.5708, 0.0, 0.0)),
    ((0.0285, 0.0, 0.105), (0.0, 1.5708, 0.0)),
    ((-0.105, 0.0, 0.0285), (0.0, -1.5708, 0.0)),
)
JOINT_LIMITS = (2.68, 2.61, 2.61, 2.6, 2.53, 2.6)

# Whiteboard station values from the handout, in base_link.
BOARD_X = 0.55
MARKER_TIP = 0.222  # marker tip ahead of end_effector_link along its z axis
WRITING_BOX_Y = (-0.25, 0.25)
WRITING_BOX_Z = (0.04, 0.30)
BOX_MARGIN = 0.05
PEN_DOWN_TOLERANCE = 0.01
MAX_KEY_POSE_GAP = 0.15
MAX_MARKER_TILT_DEG = 15.0
COURSE_POSES = {
    "RETRACT": (-0.0527, 0.3669, 2.59, -1.5359, -0.6988, -1.5189),
    "WORK": (0.40, 0.02, 2.27, -1.57, -0.84, 1.97),
    "the all-zero home pose": (0.0,) * 6,
}
MIN_POSE_DIFFERENCE = 0.1  # rad, largest single-joint difference

DH_KEYS = ("theta_offset", "d", "a", "alpha")  # the lecture's (theta, d, a, alpha) columns
FIXED_KEY = "fixed"  # marks an intermediate frame: a row with no joint variable
JOINT_COUNT = 6
CONVENTIONS = {"classical": "classical", "standard": "classical", "modified": "modified"}
# CONVENTION is optional and not in the handout: the lecture teaches the classical
# form only, which is the default. "modified" is accepted for students who use it.
REQUIRED_NAMES = ("DH", "KEY_POSES", "JACOBIAN_POSE", "JACOBIAN")
KINEMATICS_NAMES = ("CONVENTION",) + REQUIRED_NAMES


class _NotAValue(ValueError):
    pass


def _value(node: ast.AST) -> object:
    """Evaluate a literal written in kinematics.py without running the file: numbers,
    True and False, strings, lists, tuples, dicts, + - * /, [x] * n, pi (also math.pi and np.pi), and
    np.array(...) around a literal."""
    if isinstance(node, ast.Constant):
        if node.value is None:
            raise _NotAValue(f"line {node.lineno} still has None; fill in the value")
        if isinstance(node.value, (bool, int, float, str)):
            return node.value
    elif isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    elif (isinstance(node, ast.Attribute) and node.attr == "pi"
          and isinstance(node.value, ast.Name) and node.value.id in ("math", "np", "numpy")):
        return math.pi
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _number(_value(node.operand), node)
        return -operand if isinstance(node.op, ast.USub) else operand
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and any(
            isinstance(side, ast.List) for side in (node.left, node.right)):
        items, count = ((node.left, node.right) if isinstance(node.left, ast.List)
                        else (node.right, node.left))
        repeat = _value(count)
        if not isinstance(repeat, int):
            raise _NotAValue(f"line {node.lineno} repeats a list a non-whole number of times")
        return _value(items) * repeat
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _number(_value(node.left), node)
        right = _number(_value(node.right), node)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise _NotAValue(f"line {node.lineno} divides by zero")
        return left / right
    elif isinstance(node, (ast.List, ast.Tuple)):
        return [_value(element) for element in node.elts]
    elif isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
        return {_value(key): _value(value) for key, value in zip(node.keys, node.values)}
    elif (isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords
          and isinstance(node.func, ast.Attribute) and node.func.attr == "array"
          and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy")):
        return _value(node.args[0])
    raise _NotAValue(
        f"line {getattr(node, 'lineno', '?')} is not a plain value. Write numbers directly "
        "(pi, + - * / and np.array([...]) are allowed); the grader reads the file without "
        "running it."
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: object, node: ast.AST) -> float:
    if _is_number(value) and math.isfinite(value):
        return float(value)
    raise _NotAValue(f"line {node.lineno} needs a number here, not {value!r}")


def load_kinematics(repo: Path) -> tuple[dict[str, object], list[str]]:
    """Read the names in KINEMATICS_NAMES from kinematics.py without executing it."""
    path = repo / KINEMATICS_FILE
    if not path.is_file():
        return {}, [
            "Add kinematics.py at the root of your repository, as the handout's "
            "Kinematics file section describes."
        ]
    try:
        tree = ast.parse(_read_text(path), filename=str(KINEMATICS_FILE))
    except SyntaxError as exc:
        return {}, [f"kinematics.py has a Python syntax error on line {exc.lineno}: {exc.msg}."]

    values: dict[str, object] = {}
    errors: list[str] = []
    for statement in tree.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in KINEMATICS_NAMES):
            name = statement.targets[0].id
            try:
                values[name] = _value(statement.value)
            except _NotAValue as exc:
                errors.append(f"kinematics.py, {name}: {exc}.")
    missing = [name for name in REQUIRED_NAMES if name not in values]
    missing = [name for name in missing if not any(error.startswith(f"kinematics.py, {name}:")
                                                   for error in errors)]
    if missing:
        errors.append("kinematics.py does not set " + ", ".join(missing) + ".")
    return values, errors


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _origin_transform(xyz: tuple[float, ...], rpy: tuple[float, ...]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
        [-sp, cp * sr, cp * cr, xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rot_z(angle: float) -> list[list[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def gen3_lite_frames(q: list[float]) -> list[list[list[float]]]:
    """base_link to each joint's child link (shoulder_link ... end_effector_link) at q."""
    frames = []
    current = [[float(i == j) for j in range(4)] for i in range(4)]
    for (xyz, rpy), angle in zip(GEN3_LITE_JOINTS, q):
        current = _matmul(_matmul(current, _origin_transform(xyz, rpy)), _rot_z(angle))
        frames.append(current)
    return frames


def gen3_lite_fk(q: list[float]) -> list[list[float]]:
    """base_link to end_effector_link at joint values q (radians, as in /joint_states)."""
    return gen3_lite_frames(q)[-1]


def _joint_vector(name: str, value: object) -> tuple[list[float] | None, str | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        return None, f"key pose '{name}' needs six joint values, q1 to q6."
    if not all(_is_number(x) for x in value):
        return None, f"key pose '{name}' has a value that is not a number."
    q = [float(x) for x in value]
    for index, (angle, limit) in enumerate(zip(q, JOINT_LIMITS), start=1):
        if abs(angle) > limit + 1e-6:
            return None, (f"key pose '{name}': joint {index} is {angle:.3f} rad, outside its "
                          f"limits of +/-{limit} rad. Use radians, as in /joint_states.")
    return q, None


def check_key_poses(key_poses: object) -> list[str]:
    if not isinstance(key_poses, dict) or not key_poses:
        return ["KEY_POSES must be a dictionary of your three key poses: name -> [q1, ..., q6]."]
    errors: list[str] = []
    if len(key_poses) != 3:
        errors.append(f"KEY_POSES has {len(key_poses)} poses; the handout asks for three.")

    poses: dict[str, list[float]] = {}
    for name, value in key_poses.items():
        q, error = _joint_vector(str(name), value)
        if error:
            errors.append(error)
        else:
            poses[str(name)] = q

    pen_down = False
    for name, q in poses.items():
        course = [label for label, pose in COURSE_POSES.items()
                  if max(abs(a - b) for a, b in zip(q, pose)) < MIN_POSE_DIFFERENCE]
        if course:
            errors.append(f"key pose '{name}' is {course[0]}, which the course gives you. "
                          "Key poses must be poses you chose for writing.")
            continue
        frame = gen3_lite_fk(q)
        marker = [frame[i][2] for i in range(3)]
        tip = [frame[i][3] + MARKER_TIP * marker[i] for i in range(3)]
        where = f"(tip at x={tip[0]:.3f}, y={tip[1]:.3f}, z={tip[2]:.3f} m in base_link)"
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, marker[0]))))
        gap = BOARD_X - tip[0]
        if tilt > MAX_MARKER_TILT_DEG:
            errors.append(f"key pose '{name}' points the marker {tilt:.0f} degrees away from "
                          f"perpendicular to the board {where}. Key poses are poses used "
                          "while writing, with the marker facing the board.")
        elif not -PEN_DOWN_TOLERANCE <= gap <= MAX_KEY_POSE_GAP:
            errors.append(f"key pose '{name}' puts the marker tip {gap:.3f} m in front of the "
                          f"board {where}. Key poses are at the board or within "
                          f"{MAX_KEY_POSE_GAP} m of it.")
        elif not (WRITING_BOX_Y[0] - BOX_MARGIN <= tip[1] <= WRITING_BOX_Y[1] + BOX_MARGIN
                  and WRITING_BOX_Z[0] - BOX_MARGIN <= tip[2] <= WRITING_BOX_Z[1] + BOX_MARGIN):
            errors.append(f"key pose '{name}' is outside the writing box {where}.")
        elif gap <= PEN_DOWN_TOLERANCE:
            pen_down = True

    names = list(poses)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if max(abs(a - b) for a, b in zip(poses[first], poses[second])) < MIN_POSE_DIFFERENCE:
                errors.append(f"key poses '{first}' and '{second}' are the same pose.")
    if poses and not errors and not pen_down:
        errors.append("None of your key poses has the marker on the board. At least one "
                      "must be a pen-down pose, with the tip within "
                      f"{PEN_DOWN_TOLERANCE} m of the board surface.")
    return errors


def is_fixed_row(row: dict[str, object]) -> bool:
    return row.get(FIXED_KEY) is True


def check_dh(dh: object) -> list[str]:
    """One row per joint, in order, plus any intermediate frames marked "fixed": True."""
    if not isinstance(dh, list) or not dh:
        return ["DH must be a list of rows, one per joint, joint 1 first."]
    errors: list[str] = []
    for index, row in enumerate(dh, start=1):
        if not isinstance(row, dict) or not set(DH_KEYS) <= set(row) <= set(DH_KEYS) | {FIXED_KEY}:
            errors.append(f"DH row {index} must have the keys " + ", ".join(DH_KEYS)
                          + f', and "{FIXED_KEY}": True only for an intermediate frame.')
        elif not all(_is_number(row[key]) for key in DH_KEYS):
            errors.append(f"DH row {index} has a value that is not a number.")
        elif FIXED_KEY in row and not isinstance(row[FIXED_KEY], bool):
            errors.append(f'DH row {index}: "{FIXED_KEY}" must be True or False.')
    if not errors:
        joints = sum(1 for row in dh if not is_fixed_row(row))
        if joints != JOINT_COUNT:
            errors.append(f"DH has {joints} joint rows; the Gen3 Lite has {JOINT_COUNT} joints. "
                          f'Mark each intermediate frame with "{FIXED_KEY}": True.')
    return errors


def check_kinematics_file(values: dict[str, object]) -> list[str]:
    errors: list[str] = []
    convention = values.get("CONVENTION")
    if "CONVENTION" in values and str(convention).strip().lower() not in CONVENTIONS:
        errors.append("CONVENTION must be \"classical\" or \"modified\".")

    if "DH" in values:
        errors.extend(check_dh(values["DH"]))

    if "KEY_POSES" in values:
        errors.extend(check_key_poses(values["KEY_POSES"]))

    if "JACOBIAN_POSE" in values and "KEY_POSES" in values:
        key_poses = values["KEY_POSES"]
        if not isinstance(key_poses, dict) or values["JACOBIAN_POSE"] not in key_poses:
            errors.append("JACOBIAN_POSE must be the name of one of your KEY_POSES.")
    if "JACOBIAN" in values:
        jacobian = values["JACOBIAN"]
        if not (isinstance(jacobian, list) and len(jacobian) == 6
                and all(isinstance(row, list) and len(row) == 6 for row in jacobian)):
            errors.append("JACOBIAN must be 6 rows of 6 numbers (rows vx, vy, vz, wx, wy, wz; "
                          "columns joints 1 to 6).")
        elif not all(_is_number(x) for row in jacobian for x in row):
            errors.append("JACOBIAN has a value that is not a number.")
    return errors


def check_run_command(repo: Path) -> list[str]:
    """The package must provide an executable named write_initials."""
    entry = re.compile(r"['\"]\s*" + RUN_COMMAND + r"\s*=")
    for package in find_packages(repo):
        setup = package / "setup.py"
        if setup.is_file() and entry.search(_read_text(setup)):
            return []
        cmake = package / "CMakeLists.txt"
        if cmake.is_file() and RUN_COMMAND in _read_text(cmake):
            return []
    return [
        f"Your package does not provide an executable named {RUN_COMMAND}. In setup.py, "
        "add it to console_scripts, for example "
        f"'{RUN_COMMAND} = midterm_writer.write_initials:main', so that "
        f"'ros2 run <your package> {RUN_COMMAND}' starts your program."
    ]


def check_addendum(repo: Path) -> list[str]:
    values, errors = load_kinematics(repo)
    errors.extend(check_kinematics_file(values))
    errors.extend(check_run_command(repo))
    return errors


def _assert_no_errors(errors: list[str]) -> None:
    assert not errors, "\n- " + "\n- ".join(errors)


def test_repository_evidence() -> None:
    _assert_no_errors(check_required_files(REPO_ROOT))
    for notice in check_repository_hygiene(REPO_ROOT):
        warnings.warn("No points lost: " + notice)


def test_kinematics_file_and_run_command() -> None:
    _assert_no_errors(check_addendum(REPO_ROOT))
