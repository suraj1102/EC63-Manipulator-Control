"""Robot tab of plan_gui: live controller data, suction, run speed, info and raw commands."""
import json
from collections.abc import Callable

import viser

from owl_direct import ROBOT_MODE, ROBOT_STATE, EliteRPC, RpcError, StateStream, info

# The suction gripper is wired to control-box output Y1 (OWL_Arm_Field_Notes.md section 7).
SUCTION_OUTPUT = 1


def bits_on(mask: int) -> list[int]:
    return [i for i in range(64) if mask >> i & 1]


def rounded(values: list[float], digits: int = 2) -> list[float]:
    return [round(v, digits) for v in values]


def state_table(d: dict) -> str:
    """The 8056 state packet as a markdown table, like `owl_direct.py monitor` and `io`."""
    rows = [
        ("Mode", ROBOT_MODE.get(d["robotMode"], d["robotMode"])),
        ("State", ROBOT_STATE.get(d["robotState"], d["robotState"])),
        ("Servo ready", d["servoReady"]),
        ("E-stop", d["emergencyStopState"]),
        ("Collision", d["collision"]),
        ("Joints (deg)", rounded(d["machinePos"][:6])),
        ("Joint speed (deg/s)", rounded(d["joint_speed"][:6])),
        ("Flange (mm, rad)", rounded(d["machineFlangePose"], 3)),
        ("TCP speed", round(d["tcp_speed"], 2)),
        ("Torque", rounded(d["torque"][:6])),
        ("Outputs on (Y)", bits_on(d["digital_ioOutput"])),
        ("Inputs on (X)", bits_on(d["digital_ioInput"])),
        ("Analog in", rounded(d["analog_ioInput"], 3)),
        ("Analog out", rounded(d["analog_ioOutput"], 3)),
    ]
    return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


class RobotPanel:
    def __init__(self, gui: viser.GuiApi, rpc: EliteRPC, stream: StateStream, report: Callable[[Callable[[], object]], None]):
        self.rpc, self.stream, self.report = rpc, stream, report
        self.suction_on = None

        self.suction_button = gui.add_button("Suction: reading...")
        self.suction_button.on_click(lambda _: self.report(lambda: self.rpc.set_do(SUCTION_OUTPUT, not self.suction_on)))
        self.run_speed = gui.add_slider("Controller speed %", 1, 100, 1, round(rpc.speed()))
        self.run_speed.on_update(lambda _: self.report(lambda: self.rpc.set_speed(self.run_speed.value)))

        with gui.add_folder("Live data"):
            self.data = gui.add_markdown("")
        with gui.add_folder("Controller info", expand_by_default=False):
            gui.add_button("Read info").on_click(lambda _: self.read_info())
            self.info = gui.add_markdown("")
        with gui.add_folder("Raw command", expand_by_default=False):
            self.method = gui.add_text("Method", "getRobotState")
            self.params = gui.add_text("Params (JSON)", "{}")
            gui.add_button("Send").on_click(lambda _: self.send_raw())
            self.result = gui.add_markdown("")

    def update(self) -> None:
        d = self.stream.latest
        self.data.content = state_table(d)
        suction_on = bool(d["digital_ioOutput"] >> SUCTION_OUTPUT & 1)
        if suction_on == self.suction_on:
            return
        self.suction_on = suction_on
        self.suction_button.label = "Suction ON: click to release" if suction_on else "Suction OFF: click to grip"
        self.suction_button.color = "green" if suction_on else "gray"

    def read_info(self) -> None:
        self.info.content = "\n".join(f"- **{k}**: `{v}`" for k, v in info(self.rpc).items())

    def send_raw(self) -> None:
        try:
            result = self.rpc.call(self.method.value, json.loads(self.params.value))
        except (RpcError, ValueError) as e:
            result = f"error: {e}"
        self.result.content = f"```\n{json.dumps(result)}\n```"
