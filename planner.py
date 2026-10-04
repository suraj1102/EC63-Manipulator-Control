"""Collision-aware IK and trajectory planning for the OWL 6.3 (Elite EC63) with PyRoki.

Joint angles are radians and lengths metres here; owl_direct converts to the controller's degrees.
"""
import io
import math
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from jaxls import Cost
from pyroki.collision import Box, RobotCollision
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from owl_direct import EliteRPC, RpcError

# getDH on this arm, in mm (OWL_Arm_Field_Notes.md section 9).
D1, SHOULDER_Y, A2, ELBOW_Y, A3, WRIST1_Y, D5, D6 = (
    v / 1000 for v in (139.472, 119.601, 270.634, 113.899, 256.411, 98.101, 98.137, 89.870)
)
D4 = SHOULDER_Y - ELBOW_Y + WRIST1_Y

# Standard DH rows (a, alpha, d). Reproduces the controller's flange pose exactly.
DH_ROWS = [(0, -math.pi / 2, D1), (A2, 0, 0), (A3, 0, 0), (0, -math.pi / 2, D4), (0, -math.pi / 2, D5), (0, 0, D6)]

# EC63 manual limits, pulled in slightly so the controller's soft limits never trip.
JOINT_LIMITS = np.radians([355, 355, 155, 355, 355, 355])
VEL_LIMITS = np.radians([144, 144, 180, 224, 224, 224])
# The manual gives no joint acceleration limit; this reaches full speed in 0.5 s.
ACC_LIMITS = 2 * VEL_LIMITS

# Pose the arm was found in on the hardware session.
HOME = np.radians([-183.40, -90.34, -56.21, -120.77, -90.61, -15.51])

WORLD_MARGIN = 0.02
SELF_MARGIN = 0.01
TRAJ_STEPS = 50
# Controller servo period; ML trajectories are sampled at this rate.
ML_PERIOD = 0.008
# Moves smaller than this are skipped: the controller rejects a trajectory that does not move.
ALREADY_THERE = np.radians(0.01)
# Execute refuses if the arm is further than this from the plan's start.
START_TOLERANCE = np.radians(0.05)

# Link bodies as (start, end, radius) segments in each DH frame: link tubes and joint housings.
# Radii are the smallest that enclose every vertex of the owl_description owl63 meshes.
LINK_SEGMENTS = {
    "link1": [((0, -0.044, 0), (0, 0.034, 0), 0.055), ((0, 0, -0.024), (0, 0, 0.035), 0.055)],
    "link2": [
        ((-A2, 0, SHOULDER_Y), (0, 0, SHOULDER_Y), 0.052),
        ((-A2, 0, 0.084), (-A2, 0, 0.163), 0.055),
        ((0, 0, 0.084), (0, 0, 0.163), 0.055),
    ],
    "link3": [
        ((-A3, 0, SHOULDER_Y - ELBOW_Y), (0, 0, SHOULDER_Y - ELBOW_Y), 0.052),
        ((-A3, 0, -0.024), (-A3, 0, 0.035), 0.055),
        ((0, 0, -0.02), (0, 0, 0.04), 0.046),
    ],
    "link4": [((0, -0.018, 0), (0, 0.029, 0), 0.044), ((0, 0, -0.035), (0, 0, 0.029), 0.044)],
    "link5": [((0, -0.017, 0), (0, 0.029, 0), 0.044), ((0, 0, -0.035), (0, 0, 0.03), 0.044)],
    "link6": [((0, 0, -0.02), (0, 0, -0.02), 0.044)],
}

# These wrist links are always within the self-collision margin of each other.
IGNORE_PAIRS = (("link3", "link5"), ("link4", "link6"))

self_collision_constraint = Cost.factory(kind="constraint_leq_zero")(pk.costs.self_collision_residual)


class PlanError(Exception):
    pass


def dh_transform(a: float, alpha: float, d: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[1, 0, 0, a], [0, ca, -sa, 0], [0, sa, ca, d], [0, 0, 0, 1]])


def link_segments(tool_length: float, tool_radius: float) -> dict[str, list]:
    """Segments in URDF link frames, which sit one DH transform before the DH frames."""
    tool = ((0, 0, 0), (0, 0, max(tool_length - tool_radius, 0)), tool_radius)
    segments = dict(LINK_SEGMENTS, link6=LINK_SEGMENTS["link6"] + [tool])
    out = {}
    for i, (name, segs) in enumerate(segments.items()):
        T = dh_transform(*DH_ROWS[i])
        out[name] = [((T @ [*p0, 1])[:3], (T @ [*p1, 1])[:3], r) for p0, p1, r in segs]
    return out


def origin_xml(T: np.ndarray) -> str:
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return f'<origin xyz="{" ".join(map(str, T[:3, 3]))}" rpy="{" ".join(map(str, rpy))}"/>'


def frame_with_z(z: np.ndarray) -> np.ndarray:
    """Rotation whose z axis is along z and whose x axis is horizontal, so its rpy never hits gimbal lock."""
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1.0, 0, 0])
    return np.column_stack([x, np.cross(z, x), z])


def capsule_visual_xml(p0: np.ndarray, p1: np.ndarray, r: float) -> str:
    length = float(np.linalg.norm(p1 - p0))
    T = np.eye(4)
    T[:3, 3] = (p0 + p1) / 2
    if length > 0:
        T[:3, :3] = frame_with_z(p1 - p0)
    ends = "".join(
        f'<visual><origin xyz="{" ".join(map(str, p))}"/><geometry><sphere radius="{r}"/></geometry><material name="grey"/></visual>'
        for p in (p0, p1)
    )
    return f'<visual>{origin_xml(T)}<geometry><cylinder radius="{r}" length="{length}"/></geometry><material name="grey"/></visual>{ends}'


def build_urdf(segments: dict[str, list], tool_length: float) -> yourdfpy.URDF:
    parts = [
        '<robot name="owl63"><material name="grey"><color rgba="0.8 0.8 0.82 1"/></material>',
        '<link name="base_link"><visual><origin xyz="0 0 0.045"/><geometry><cylinder radius="0.065" length="0.09"/></geometry>'
        '<material name="grey"/></visual></link>',
    ]
    parent, T_parent = "base_link", np.eye(4)
    for i, name in enumerate(segments):
        visuals = "".join(capsule_visual_xml(p0, p1, r) for p0, p1, r in segments[name])
        parts.append(
            f'<link name="{name}">{visuals}</link>'
            f'<joint name="joint{i + 1}" type="revolute"><parent link="{parent}"/><child link="{name}"/>{origin_xml(T_parent)}'
            f'<axis xyz="0 0 1"/><limit lower="{-JOINT_LIMITS[i]}" upper="{JOINT_LIMITS[i]}" velocity="{VEL_LIMITS[i]}" effort="100"/></joint>'
        )
        parent, T_parent = name, dh_transform(*DH_ROWS[i])
    T_tool = T_parent @ dh_transform(0, 0, tool_length)
    parts.append(f'<link name="tool0"/><joint name="tool_joint" type="fixed"><parent link="link6"/><child link="tool0"/>{origin_xml(T_tool)}</joint>')
    parts.append("</robot>")
    return yourdfpy.URDF.load(io.StringIO("".join(parts)))


def segment_spheres(segments: dict[str, list]) -> dict[str, dict[str, list]]:
    """Fill each segment with spheres whose union contains its capsule."""
    out = {}
    for name, segs in segments.items():
        centers, radii = [], []
        for p0, p1, r in segs:
            n = max(2, math.ceil(np.linalg.norm(p1 - p0) / (r / 2)) + 1)
            spacing = np.linalg.norm(p1 - p0) / (n - 1)
            centers += np.linspace(p0, p1, n).tolist()
            radii += [math.hypot(r, spacing / 2)] * n
        out[name] = {"centers": centers, "radii": radii}
    return out


def load_boxes(boxes: list[dict]) -> Box:
    wxyz = [Rotation.from_euler("z", b.get("yaw", 0.0), degrees=True).as_quat(scalar_first=True) for b in boxes]
    return Box.from_extent(
        extent=np.array([b["size"] for b in boxes]),
        position=np.array([b["center"] for b in boxes]),
        wxyz=np.array(wxyz),
    )


class Planner:
    def __init__(self, scene_path: Path):
        self.scene = tomllib.loads(Path(scene_path).read_text())
        tool = self.scene["tool"]
        segments = link_segments(tool["length"], tool["radius"])
        self.urdf = build_urdf(segments, tool["length"])
        self.robot = pk.Robot.from_urdf(self.urdf, default_joint_cfg=HOME)
        self.robot_coll = RobotCollision.from_sphere_decomposition(segment_spheres(segments), self.urdf, IGNORE_PAIRS)
        self.world = load_boxes(self.scene["box"])
        self.tool_index = self.robot.links.names.index("tool0")

    def fk(self, q: np.ndarray) -> jaxlie.SE3:
        return jaxlie.SE3(self.robot.forward_kinematics(jnp.asarray(q))[..., self.tool_index, :])

    def ik(self, position: np.ndarray, wxyz: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Collision-free joints that put tool0 closest to the target, starting the search from seed."""
        target = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.asarray(wxyz)), jnp.asarray(position))
        return np.array(_solve_ik(self.robot, self.robot_coll, self.world, self.tool_index, target, jnp.asarray(seed)))

    def clearances(self, q: np.ndarray) -> tuple[float, float]:
        """Smallest robot-to-world and robot-to-itself distances over a batch of configurations."""
        world, own = _clearances(self.robot, self.robot_coll, self.world, jnp.atleast_2d(jnp.asarray(q)))
        return float(world), float(own)

    def clearance(self, q: np.ndarray) -> float:
        return min(self.clearances(q))

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, speed: float) -> tuple[np.ndarray, np.ndarray]:
        """Collision-free trajectory sampled at the controller period; speed scales the joint limits (0..1]."""
        if np.abs(np.asarray(q_goal) - q_start).max() < ALREADY_THERE:
            raise PlanError("already at the goal")
        for label, q in (("start", q_start), ("goal", q_goal)):
            if self.clearance(q) < 0:
                raise PlanError(f"{label} is in collision")
        # An end closer than a margin (a pick pose near the table) caps it, or the path would bend away and back.
        world_end, self_end = np.minimum(self.clearances(q_start), self.clearances(q_goal))
        margins = jnp.array([min(WORLD_MARGIN, world_end), min(SELF_MARGIN, self_end)])
        waypoints = np.array(
            _solve_trajopt(self.robot, self.robot_coll, self.world, jnp.asarray(q_start), jnp.asarray(q_goal), margins, TRAJ_STEPS)
        )
        # The optimizer pins the ends only to ~0.03 deg; a sequence's next move starts exactly where this one ends.
        waypoints[0], waypoints[-1] = q_start, q_goal
        times, q = retime(waypoints, speed)
        clearance = self.clearance(q)
        if clearance < 0:
            raise PlanError(f"no collision-free path found (penetration {-clearance * 1000:.0f} mm)")
        return times, q

    def plan_sequence(self, qs: list[np.ndarray], speed: float) -> tuple[np.ndarray, np.ndarray]:
        """Rest-to-rest moves through each configuration in turn, joined into one trajectory."""
        qs = [q for i, q in enumerate(qs) if i == 0 or np.abs(np.asarray(q) - qs[i - 1]).max() >= ALREADY_THERE]
        if len(qs) == 1:
            raise PlanError("already at every position")
        times, q = [np.zeros(1)], [np.asarray(qs[0])[None]]
        for i, (a, b) in enumerate(zip(qs, qs[1:])):
            try:
                t, segment = self.plan(a, b, speed)
            except PlanError as e:
                raise PlanError(f"move {i + 1}: {e}") from e
            times.append(t[1:] + times[-1][-1])
            q.append(segment[1:])
        return np.concatenate(times), np.concatenate(q)


def retime(waypoints: np.ndarray, speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Move along the path with a trapezoidal speed profile, as fast as speed * the joint limits allow."""
    path = CubicSpline(np.linspace(0, 1, len(waypoints)), waypoints)
    vmax, amax = speed * VEL_LIMITS, speed * ACC_LIMITS
    # Floor avoids dividing by zero when the start and goal are the same.
    dq = np.maximum(np.abs(path(np.linspace(0, 1, 1000), 1)).max(axis=0), 1e-6)
    v, a = (vmax / dq).min(), (amax / dq).min()
    # Path curvature adds acceleration the profile does not see; slow the whole move until it fits.
    _, s, ds, dds = trapezoid(v, a)
    qd = path(s, 1) * ds[:, None]
    qdd = path(s, 2) * ds[:, None] ** 2 + path(s, 1) * dds[:, None]
    k = max(1.0, (np.abs(qd).max(axis=0) / vmax).max(), math.sqrt((np.abs(qdd).max(axis=0) / amax).max()))
    times, s, _, _ = trapezoid(v / k, a / k**2)
    return times, path(s)


def trapezoid(v: float, a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rest-to-rest move over a unit distance sampled at ML_PERIOD: times, position, velocity, acceleration."""
    t_ramp = min(v / a, math.sqrt(1 / a))
    duration = t_ramp + 1 / (a * t_ramp)
    # Stretch so the move ends exactly on a sample.
    stretch = math.ceil(duration / ML_PERIOD) * ML_PERIOD / duration
    a, t_ramp, duration = a / stretch**2, t_ramp * stretch, duration * stretch
    v = a * t_ramp
    t = np.arange(round(duration / ML_PERIOD) + 1) * ML_PERIOD
    phase = [t < t_ramp, t > duration - t_ramp]
    s = np.select(phase, [a * t**2 / 2, 1 - a * (duration - t) ** 2 / 2], a * t_ramp**2 / 2 + v * (t - t_ramp))
    ds = np.select(phase, [a * t, a * (duration - t)], v)
    dds = np.select(phase, [a, -a], 0.0)
    return t, s, ds, dds


def execute(rpc: EliteRPC, times: np.ndarray, q: np.ndarray) -> None:
    current_deg = rpc.joints()[:6]
    if np.abs(np.radians(current_deg) - q[0]).max() > START_TOLERANCE:
        raise RpcError("robot is not at the start of the plan; replan from its current position")
    # The controller checks the first point against its own reading; planning in float32 leaves ~1e-5 deg.
    q_deg = np.degrees(q)
    q_deg[0] = current_deg
    rpc.run_trajectory(times, q_deg)


@jdc.jit
def _clearances(robot: pk.Robot, robot_coll: RobotCollision, world: Box, q: jax.Array) -> tuple[jax.Array, jax.Array]:
    world_dist = jax.vmap(lambda c: robot_coll.compute_world_collision_distance(robot, c, world))(q)
    self_dist = robot_coll.compute_self_collision_distance(robot, q)
    return world_dist.min(), self_dist.min()


@jdc.jit
def _solve_ik(
    robot: pk.Robot, robot_coll: RobotCollision, world: Box, tool_index: jdc.Static[int], target: jaxlie.SE3, seed: jax.Array
) -> jax.Array:
    var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost(robot, var, target, jnp.array(tool_index), pos_weight=50.0, ori_weight=10.0),
        pk.costs.rest_cost(var, seed, weight=0.01),
        pk.costs.limit_constraint(robot, var),
        self_collision_constraint(robot, robot_coll, var, SELF_MARGIN),
        pk.costs.world_collision_constraint(robot, robot_coll, var, world, WORLD_MARGIN),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[var])
        .analyze()
        .solve(initial_vals=jaxls.VarValues.make([var.with_value(seed)]), verbose=False, linear_solver="dense_cholesky")
    )
    return sol[var]


@jdc.jit
def _solve_trajopt(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    world: Box,
    q_start: jax.Array,
    q_goal: jax.Array,
    margins: jax.Array,
    steps: jdc.Static[int],
) -> jax.Array:
    world_margin, self_margin = margins
    var = robot.joint_var_cls
    traj = var(jnp.arange(steps))
    batched = lambda x: jax.tree.map(lambda leaf: leaf[None], x)
    robot_b, coll_b = batched(robot), batched(robot_coll)

    @Cost.factory(kind="constraint_eq_zero")
    def pinned(vals: jaxls.VarValues, v: jaxls.Var, q: jax.Array) -> jax.Array:
        return vals[v] - q

    def swept_clearance(vals, robot, robot_coll, world, prev, curr):
        capsules = robot_coll.get_swept_capsules(robot, vals[prev], vals[curr])
        return pk.collision.collide(capsules.reshape((-1, 1)), world.reshape((1, -1))).flatten() - world_margin

    costs = [
        pk.costs.smoothness_cost(var(jnp.arange(1, steps)), var(jnp.arange(steps - 1)), jnp.array([1.0])[None]),
        pk.costs.five_point_acceleration_cost(
            var(jnp.arange(2, steps - 2)), var(jnp.arange(4, steps)), var(jnp.arange(3, steps - 1)),
            var(jnp.arange(1, steps - 3)), var(jnp.arange(steps - 4)), 0.1, jnp.array([0.1])[None],
        ),
        pk.costs.limit_constraint(robot_b, traj),
        self_collision_constraint(robot_b, coll_b, traj, self_margin),
        pinned(var(jnp.arange(1)), q_start[None]),
        pinned(var(jnp.arange(steps - 1, steps)), q_goal[None]),
        Cost(
            swept_clearance,
            (robot_b, coll_b, batched(world), var(jnp.arange(steps - 1)), var(jnp.arange(1, steps))),
            kind="constraint_geq_zero",
        ),
    ]
    init = jaxls.VarValues.make([traj.with_value(jnp.linspace(q_start, q_goal, steps))])
    sol = jaxls.LeastSquaresProblem(costs=costs, variables=[traj]).analyze().solve(initial_vals=init, verbose=False)
    return sol[traj]
