#!/usr/bin/env python3
"""Browser motion planner for the OWL arm: drag the target, plan around the scene, execute.

  python plan_gui.py                       # simulation only
  python plan_gui.py --ip 192.168.1.200    # live robot (pendant key on REMOTE)

Open http://localhost:8080. "Show collision model" overlays the capsules the planner checks.
"""
import argparse
import io
import itertools
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import jaxlie
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from owl_direct import EliteRPC, RpcError, StateStream
from planner import ALREADY_THERE, HOME, Planner, PlanError, execute
from robot_panel import SUCTION_OUTPUT, RobotPanel

GOAL_COLOR = (0.3, 0.8, 0.4, 0.5)
COLLISION_COLOR = (0.9, 0.5, 0.2, 0.4)
POLL_PERIOD = 0.5
MESH_URDF = Path(__file__).with_name("owl_description") / "urdf" / "owl63_description.urdf"
POSITIONS_FILE = Path(__file__).with_name("positions.json")
WAIT_STEP = re.compile(r"wait\((\d+)\)")


def load_mesh_urdf(tool_length: float, tool_radius: float) -> yourdfpy.URDF:
    """The owl63 meshes with the suction tool drawn on the flange."""
    root = MESH_URDF.parents[1]
    tool = (
        f'<link name="tool"><visual><origin xyz="0 0 {tool_length / 2}"/>'
        f'<geometry><cylinder radius="{tool_radius}" length="{tool_length}"/></geometry></visual></link>'
        '<joint name="tool_joint" type="fixed"><parent link="flan"/><child link="tool"/></joint>'
    )
    xml = MESH_URDF.read_text().replace("</robot>", tool + "</robot>")
    return yourdfpy.URDF.load(
        io.BytesIO(xml.encode()), filename_handler=lambda fname: str(root / fname.removeprefix("package://owl_description/"))
    )


def load_positions() -> dict:
    """Saved joint positions in degrees by name, and the last sequence played."""
    saved = json.loads(POSITIONS_FILE.read_text()) if POSITIONS_FILE.exists() else {"positions": {}, "sequence": []}
    saved["positions"].setdefault("home", np.degrees(HOME).round(3).tolist())
    return saved


def same_joints(a: list[float], b: list[float]) -> bool:
    return np.allclose(a, b, atol=0.01)


def save_positions(saved: dict) -> None:
    POSITIONS_FILE.write_text(json.dumps(saved, indent=2))


def parse_steps(text: str) -> list[tuple[str, object]]:
    """Sequence text as ("position", name), ("suction", on) and ("wait", seconds) steps."""
    steps = []
    for token in (t.strip() for t in text.split(",")):
        if token in ("SUCK", "UNSUCK"):
            steps.append(("suction", token == "SUCK"))
        elif match := WAIT_STEP.fullmatch(token):
            steps.append(("wait", int(match[1]) / 1000))
        elif token:
            steps.append(("position", token))
    return steps


def describe(kind: str, value: object) -> str:
    if kind == "suction":
        return "SUCK" if value else "UNSUCK"
    if kind == "wait":
        return f"wait({value * 1000:.0f})"
    return "moving"


def join_moves(items: list[tuple[str, object]]) -> tuple[np.ndarray, np.ndarray] | None:
    """The motion in a list of program items as one trajectory, for the preview."""
    times, q = [], []
    for kind, value in items:
        if kind == "move":
            t, segment = value
            times.append(t + (times[-1][-1] if times else 0.0))
            q.append(segment)
    return (np.concatenate(times), np.concatenate(q)) if times else None


def format_xyz(p: np.ndarray) -> str:
    return "x {:.3f}  y {:.3f}  z {:.3f} m".format(*p)


class App:
    def __init__(self, scene_path: Path, ip: str | None):
        self.scene_path = scene_path
        self.rpc = EliteRPC(ip) if ip else None
        self.stream = StateStream(ip) if ip else None
        if self.stream:
            self.stream.wait_first()
        self.sim_q = HOME
        self.goal = HOME
        self.plan = None
        self.preview = None
        self.path = None
        self.stop_requested = False
        self.goal_reachable = True
        self.servo_enabled = None
        self.last_target = None
        self.reload_requested = True
        self.saved = load_positions()

        self.server = viser.ViserServer()
        gui = self.server.gui
        self.status = gui.add_markdown("")
        self.readout = gui.add_markdown("")
        self.speed = gui.add_slider("Speed %", 5, 100, 5, 20)
        self.execute_button = gui.add_button("Execute", disabled=True)
        self.execute_button.on_click(lambda _: self.on_execute())
        gui.add_button("Stop", color="red").on_click(lambda _: self.on_stop())
        if self.rpc:
            self.servo_button = gui.add_button("Servo: reading...")
            self.servo_button.on_click(lambda _: self.on_servo())

        tabs = gui.add_tab_group()
        with tabs.add_tab("Motion"):
            gui.add_button("Plan").on_click(lambda _: self.on_plan())
            self.show_collision = gui.add_checkbox("Show collision model", False)
            self.show_collision.on_update(lambda _: setattr(self.collision_vis, "show_visual", self.show_collision.value))
            gui.add_button("Reload scene").on_click(lambda _: setattr(self, "reload_requested", True))
        with tabs.add_tab("Positions"):
            gui.add_button("Home").on_click(lambda _: self.go_to("home"))
            self.position_name = gui.add_text("Name", "pick", hint='Saving as "home" changes the home position.')
            gui.add_button("Save arm position").on_click(lambda _: self.save_position(self.current()))
            gui.add_button("Save goal (green ghost)").on_click(lambda _: self.save_position(self.goal))
            self.position_list = gui.add_dropdown("Saved", list(self.saved["positions"]))
            gui.add_button("Go to").on_click(lambda _: self.go_to(self.position_list.value))
            gui.add_button("Delete").on_click(lambda _: self.delete_position())
            self.sequence = gui.add_text(
                "Sequence", ", ".join(self.saved["sequence"]), hint="Saved names, SUCK, UNSUCK and wait(ms), separated by commas."
            )
            self.cycles = gui.add_number("Cycles", 1, min=1, step=1)
            self.loop_forever = gui.add_checkbox("Loop forever", False, hint="Runs until Stop.")
            gui.add_button("Plan sequence").on_click(lambda _: self.on_plan_sequence())
        if self.rpc:
            with tabs.add_tab("Robot"):
                self.robot_panel = RobotPanel(gui, self.rpc, self.stream, self.report)

    def current(self) -> np.ndarray:
        if self.stream:
            # A dead reader thread leaves the last packet in `latest`, which would look like a stopped arm.
            if not self.stream.thread.is_alive():
                raise ConnectionError("state stream from the arm (port 8056) stopped")
            return np.radians(self.stream.latest["machinePos"][:6])
        return self.sim_q

    def load_scene(self) -> None:
        self.planner = Planner(self.scene_path)
        scene = self.server.scene
        scene.reset()
        # reset() already removed the old path node.
        self.path = None
        self.clear_plan()
        scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
        # Hidden until toggled with the eye icon in viser's settings menu (Scene tree > guides).
        scene.add_frame("/guides", show_axes=False, visible=False)
        scene.add_frame("/guides/axes", axes_length=0.3, axes_radius=0.006)
        for axis, tip in zip("xyz", np.eye(3) * 0.33):
            scene.add_label(f"/guides/axes/{axis}", axis, position=tip, anchor="center-center")
        for box in self.planner.scene["box"]:
            wxyz = jaxlie.SO3.from_z_radians(np.radians(box.get("yaw", 0.0))).wxyz
            scene.add_box(f"/scene/{box['name']}", (170, 120, 80), box["size"], opacity=0.6, position=box["center"], wxyz=wxyz)
            top = np.add(box["center"], (0, 0, box["size"][2] / 2))
            label = f"{box['name']}: center {box['center']}"
            scene.add_label(f"/guides/{box['name']}", label, position=top, anchor="bottom-center")
        tool = self.planner.scene["tool"]
        mesh_urdf = load_mesh_urdf(tool["length"], tool["radius"])
        self.robot_vis = ViserUrdf(self.server, mesh_urdf, root_node_name="/robot")
        self.goal_vis = ViserUrdf(self.server, mesh_urdf, root_node_name="/goal", mesh_color_override=GOAL_COLOR)
        self.collision_vis = ViserUrdf(self.server, self.planner.urdf, root_node_name="/collision", mesh_color_override=COLLISION_COLOR)
        self.collision_vis.show_visual = self.show_collision.value
        self.goal = self.current()
        tool = self.planner.fk(self.goal)
        self.target = scene.add_transform_controls("/target", scale=0.15, position=tool.translation(), wxyz=tool.rotation().wxyz)
        self.status.content = f"Loaded `{self.scene_path}`. Drag the target, then Plan."

    def show_positions(self) -> None:
        tip = self.planner.fk(self.current()).translation()
        text = f"Tool tip `{format_xyz(tip)}`  \nTarget `{format_xyz(self.target.position)}`"
        if text != self.readout.content:
            self.readout.content = text

    def update_servo(self) -> None:
        enabled = self.rpc.servo_status()
        if enabled == self.servo_enabled:
            return
        self.servo_enabled = enabled
        self.servo_button.label = "Servo ON" if enabled else "Servo OFF: click to enable"
        self.servo_button.color = "green" if enabled else "orange"

    def on_servo(self) -> None:
        if not self.servo_enabled:
            self.report(self.rpc.servo_on)

    def solve_goal(self) -> None:
        position, wxyz = np.array(self.target.position), np.array(self.target.wxyz)
        self.goal = self.planner.ik(position, wxyz, self.goal)
        self.clear_plan()
        reached = self.planner.fk(self.goal)
        pos_err = np.linalg.norm(reached.translation() - position) * 1000
        ang_err = np.degrees(np.linalg.norm((reached.rotation().inverse() @ jaxlie.SO3(wxyz)).log()))
        self.goal_reachable = pos_err < 1 and ang_err < 1
        reachable = "reachable" if self.goal_reachable else "**not reachable without collision**"
        self.status.content = f"Goal {reachable}: error {pos_err:.1f} mm, {ang_err:.1f} deg"

    def set_goal_joints(self, q: np.ndarray) -> None:
        """Make q the goal and put the target on it, without re-solving IK."""
        self.goal = np.asarray(q)
        self.goal_reachable = True
        self.clear_plan()
        tool = self.planner.fk(self.goal)
        self.target.position, self.target.wxyz = np.array(tool.translation()), np.array(tool.rotation().wxyz)
        self.last_target = (tuple(self.target.position), tuple(self.target.wxyz))

    def go_to(self, name: str) -> None:
        self.set_goal_joints(np.radians(self.saved["positions"][name]))
        self.on_plan()

    def save_position(self, q: np.ndarray) -> None:
        name = self.position_name.value.strip()
        joints = np.degrees(np.asarray(q, dtype=float)).round(3).tolist()
        self.saved["positions"][name] = joints
        save_positions(self.saved)
        self.position_list.options = list(self.saved["positions"])
        self.position_list.value = name
        twins = [n for n, other in self.saved["positions"].items() if n != name and same_joints(other, joints)]
        warning = f" **Same joints as {', '.join(twins)}.**" if twins else ""
        self.status.content = f"Saved `{name}`: {joints}.{warning}"

    def delete_position(self) -> None:
        name = self.position_list.value
        if name == "home":
            self.status.content = "`home` can be overwritten but not deleted."
            return
        del self.saved["positions"][name]
        save_positions(self.saved)
        self.position_list.options = list(self.saved["positions"])
        self.status.content = f"Deleted `{name}`."

    def on_plan_sequence(self) -> None:
        steps = parse_steps(self.sequence.value)
        names = [value for kind, value in steps if kind == "position"]
        positions = self.saved["positions"]
        missing = [n for n in names if n not in positions]
        if not steps or missing:
            self.status.content = f"Unknown positions: {missing}" if missing else "Type a sequence first."
            return
        twins = [(a, b) for a, b in zip(names, names[1:]) if a != b and same_joints(positions[a], positions[b])]
        if twins:
            self.status.content = f"`{twins[0][0]}` and `{twins[0][1]}` are saved with the same joints. Save one of them again."
            return
        self.saved["sequence"] = [t.strip() for t in self.sequence.value.split(",") if t.strip()]
        save_positions(self.saved)
        cycles = None if self.loop_forever.value else int(self.cycles.value)
        speed = self.speed.value / 100

        def build() -> tuple[list, list, int | None]:
            first, end = self.plan_steps(self.current(), steps, speed)
            # Later cycles start where the sequence ends, not where the arm is now.
            repeat = self.plan_steps(end, steps, speed)[0] if cycles != 1 else []
            return first, repeat, cycles

        self.run_planner(build)

    def plan_steps(self, start: np.ndarray, steps: list[tuple[str, object]], speed: float) -> tuple[list, np.ndarray]:
        """Program items for one pass of the steps from start, and the configuration it ends at."""
        items, prev = [], start
        for is_position, group in itertools.groupby(steps, key=lambda step: step[0] == "position"):
            group = list(group)
            if not is_position:
                items += group
                continue
            qs = [prev] + [np.radians(self.saved["positions"][name]) for _, name in group]
            # A stretch the arm is already at is skipped, such as the first position when a cycle repeats from it.
            if any(np.abs(q - prev).max() >= ALREADY_THERE for q in qs[1:]):
                items.append(("move", self.planner.plan_sequence(qs, speed)))
            prev = qs[-1]
        return items, prev

    def on_plan(self) -> None:
        if not self.goal_reachable:
            self.status.content = "Goal not reachable without collision. Move the target."
            return
        self.run_planner(lambda: ([("move", self.planner.plan(self.current(), self.goal, self.speed.value / 100))], [], 1))

    def run_planner(self, build: Callable[[], tuple[list, list, int | None]]) -> None:
        self.status.content = "Planning..."
        try:
            program = build()
        except PlanError as e:
            self.status.content = f"Plan failed: {e}"
            return
        self.plan = program
        self.preview = join_moves(program[0])
        self.plan_start_time = time.time()
        self.execute_button.disabled = False
        cycles = program[2]
        runs = "" if cycles == 1 else " per cycle, looping forever" if cycles is None else f" per cycle, {cycles} cycles"
        if not self.preview:
            self.status.content = f"Planned: no motion{runs}. Execute when ready."
            return
        times, q = self.preview
        self.path = self.server.scene.add_spline_catmull_rom("/plan_path", self.planner.fk(q[::25]).translation(), color=(40, 160, 80), line_width=3)
        self.status.content = f"Planned {times[-1]:.1f} s of motion{runs}, clearance {self.planner.clearance(q) * 1000:.0f} mm. Execute when ready."

    def on_execute(self) -> None:
        first, repeat, cycles = self.plan
        self.clear_plan()
        self.stop_requested = False
        try:
            self.run_program(first, repeat, cycles)
        except RpcError as e:
            self.status.content = f"Robot: {e}"
        # Where the arm actually ended up, which differs from the plan's end if it was stopped.
        self.set_goal_joints(self.current())

    def run_program(self, first: list, repeat: list, cycles: int | None) -> None:
        cycle = 0
        while cycles is None or cycle < cycles:
            of = "" if cycles is None else f" of {cycles}"
            for kind, value in first if cycle == 0 else repeat:
                if self.stop_requested:
                    self.status.content = f"Stopped in cycle {cycle + 1}{of}."
                    return
                self.status.content = f"Cycle {cycle + 1}{of}: {describe(kind, value)}"
                self.run_item(kind, value)
            cycle += 1
        self.status.content = f"Done: {cycle} cycle{'s' if cycle > 1 else ''}."

    def run_item(self, kind: str, value: object) -> None:
        if kind == "move":
            times, q = value
            if self.rpc:
                execute(self.rpc, times, q)
            else:
                self.animate(times, q)
        elif kind == "suction":
            if self.rpc:
                self.rpc.set_do(SUCTION_OUTPUT, value)
        else:
            deadline = time.time() + value
            while time.time() < deadline and not self.stop_requested:
                time.sleep(0.02)

    def animate(self, times: np.ndarray, q: np.ndarray) -> None:
        t0 = time.time()
        while (t := time.time() - t0) < times[-1] and not self.stop_requested:
            self.sim_q = q[np.searchsorted(times, t)]
            time.sleep(0.02)
        if not self.stop_requested:
            self.sim_q = q[-1]

    def on_stop(self) -> None:
        self.stop_requested = True
        self.status.content = "Stopping..."
        if self.rpc:
            try:
                self.rpc.stop()
            except RpcError as e:
                self.status.content = f"Robot: {e}"

    def clear_plan(self) -> None:
        self.plan = None
        self.preview = None
        self.execute_button.disabled = True
        if self.path:
            self.path.remove()
            self.path = None

    def report(self, action: Callable[[], object]) -> None:
        try:
            action()
            self.status.content = "Done."
        except RpcError as e:
            self.status.content = f"Robot: {e}"

    def run(self) -> None:
        last_poll = 0.0
        while True:
            if self.rpc and time.time() - last_poll > POLL_PERIOD:
                last_poll = time.time()
                self.update_servo()
                self.robot_panel.update()
            if self.reload_requested:
                self.reload_requested = False
                self.load_scene()
                self.last_target = None
            target = (tuple(self.target.position), tuple(self.target.wxyz))
            if target != self.last_target:
                self.last_target = target
                self.solve_goal()
            self.robot_vis.update_cfg(self.current())
            self.show_positions()
            self.collision_vis.update_cfg(self.current())
            if self.preview:
                times, q = self.preview
                # Loop the planned motion on the goal ghost, like MoveIt's trajectory preview.
                t = (time.time() - self.plan_start_time) % (times[-1] + 1)
                self.goal_vis.update_cfg(q[min(np.searchsorted(times, t), len(q) - 1)])
            else:
                self.goal_vis.update_cfg(self.goal)
            time.sleep(0.03)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", help="controller address; omit to simulate")
    ap.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene.toml"))
    a = ap.parse_args()
    App(a.scene, a.ip).run()


if __name__ == "__main__":
    main()
