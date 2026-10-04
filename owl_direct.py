#!/usr/bin/env python3
"""
owl_direct.py - talk to an Elite Robots EC63/EC66/EC612 controller directly. No
ROS, no MoveIt, no `elirobots` SDK. Python 3 stdlib only.

Reverse-engineered from owl_robot_ros + the elirobots 0.0.7 SDK it depends on.

  TCP 8055  JSON-RPC 2.0 command port   (one JSON object per line, request/response)
  TCP 8056  real-time state stream      (binary, big-endian, one packet every 8 ms)

Units used by the controller:
  joints         degrees, ALWAYS 8 values (6 arm + 2 external axes; pad with 0)
  cartesian pose [x, y, z, rx, ry, rz]  x/y/z in mm, rx/ry/rz in rad (unless unit_type=0 -> deg)
  joint speed    % (1..100) for moveByJoint; mm/s for moveByLine

Usage examples (robot must be in REMOTE mode on the teach pendant for motion):
  python3 owl_direct.py --ip 192.168.1.200 monitor
  python3 owl_direct.py --ip 192.168.1.200 info
  python3 owl_direct.py --ip 192.168.1.200 servo_on
  python3 owl_direct.py --ip 192.168.1.200 movej 0 -90 90 -90 90 0 --speed 10
  python3 owl_direct.py --ip 192.168.1.200 checkfk --robot owl66
  python3 owl_direct.py --ip 192.168.1.200 raw getRobotState
  python3 owl_direct.py --ip 192.168.1.200 raw get_joint_pos
"""
import argparse
import json
import math
import socket
import struct
import threading
import time

CMD_PORT = 8055
STATE_PORT = 8056

ROBOT_MODE = {0: "TEACH", 1: "PLAY", 2: "REMOTE"}
ROBOT_STATE = {0: "STOP", 1: "PAUSE", 2: "ESTOP", 3: "PLAY(running)", 4: "ERROR", 5: "COLLISION"}
ML_PUSH_RESULT = {0: "CORRECT", -1: "WRONG_LENGTH", -2: "WRONG_FORMAT", -3: "TIMESTAMP_IS_NOT_STANDARD"}


class RpcError(Exception):
    pass


# --------------------------------------------------------------------------- 8055
class EliteRPC:
    """JSON-RPC client for port 8055.

    Wire format (request, newline terminated):
        {"method":"<name>","params":{...},"jsonrpc":"2.0","id":<n>}\n
    Response:
        {"jsonrpc":"2.0","result":"<JSON ENCODED AS A STRING>","id":<n>}
        {"jsonrpc":"2.0","error":{"code":..,"message":".."},"id":<n>}
    Note `result` is itself a JSON string ("true", "[0.1, ...]", "3") that must be
    decoded a second time.
    """

    def __init__(self, ip, port=CMD_PORT, timeout=5.0):
        self.sock = socket.create_connection((ip, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._lock = threading.Lock()
        self._id = 0
        self._buf = b""

    def close(self):
        self.sock.close()

    def _read_json(self):
        dec = json.JSONDecoder()
        while True:
            text = self._buf.decode("utf-8", errors="replace").lstrip()
            if text:
                try:
                    obj, end = dec.raw_decode(text)
                    self._buf = text[end:].encode("utf-8")
                    return obj
                except ValueError:
                    pass
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("controller closed 8055 connection")
            self._buf += chunk

    def call(self, method, params=None, expect_reply=True):
        with self._lock:
            self._id += 1
            req = {"method": method, "params": params or {}, "jsonrpc": "2.0", "id": self._id}
            self.sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
            if not expect_reply:
                return None
            resp = self._read_json()
        if "error" in resp:
            raise RpcError(f"{method}: {resp['error'].get('message', resp['error'])}")
        res = resp.get("result")
        if isinstance(res, str):
            try:
                return json.loads(res)
            except ValueError:
                return res
        return res

    # ---- state / info -------------------------------------------------------
    def mode(self):            return self.call("getRobotMode")          # 0 teach 1 play 2 remote
    def state(self):           return self.call("getRobotState")         # 0 stop .. 5 collision
    def servo_status(self):    return self.call("getServoStatus")
    def sync_status(self):     return self.call("getMotorStatus")
    def estop(self):           return self.call("get_estop_status")
    def joints(self):          return self.call("get_joint_pos")         # deg, 8 values (fw >= 2.19.2)
    def tcp_pose(self):        return self.call("get_tcp_pose", {"coordinate_num": -1, "tool_num": -1})
    def flange_pose(self):     return self.call("get_base_flange_pose")
    def version(self):         return self.call("getSoftVersion")
    def subtype(self):         return self.call("getRobotSubtype")       # 3=EC63 6=EC66 12=EC612
    def alarm(self):           return self.call("getAlarmNum")
    def speed(self):           return self.call("getSpeed")              # run-speed override, %
    def set_speed(self, pct):  return self.call("setSpeed", {"value": pct})  # 0.05..100
    def dh(self):              return [self.call("getDH", {"index": i}) for i in range(11)]
    def fk(self, j8, unit_type=None):
        p = {"targetPos": pad8(j8)}
        if unit_type is not None: p["unit_type"] = unit_type
        return self.call("positiveKinematic", p)
    def ik(self, pose, ref_j8=None, unit_type=None):
        p = {"targetPose": list(pose)}
        if ref_j8 is not None: p["referencePos"] = pad8(ref_j8)
        if unit_type is not None: p["unit_type"] = unit_type
        return self.call("inverseKinematic", p)

    # ---- power-up sequence (what elite.EC.robot_servo_on does) --------------
    def servo_on(self, timeout=10.0):
        m = self.mode()
        if m != 2:
            raise RpcError(f"robot is in {ROBOT_MODE.get(m, m)} mode - switch the pendant key to REMOTE")
        for _ in range(5):
            self.call("clearAlarm")
            time.sleep(0.2)
            if self.state() == 0:
                break
        else:
            raise RpcError(f"cannot clear alarm, state={ROBOT_STATE.get(self.state())}, alarm={self.alarm()}")
        if not self.sync_status():
            self.call("syncMotorStatus")
            time.sleep(0.2)
        t0 = time.time()
        while not self.servo_status():
            self.call("set_servo_status", {"status": 1})
            time.sleep(0.05)
            if time.time() - t0 > timeout:
                raise RpcError("servo did not enable")
        return True

    def servo_off(self):
        return self.call("set_servo_status", {"status": 0})

    # ---- motion ---------------------------------------------------------------
    def stop(self):            return self.call("stop")
    def pause(self):           return self.call("pause")
    def resume(self):          return self.call("run")

    def wait_motion(self, start_timeout=1.0, timeout=120.0):
        """Wait for a motion to START and then FINISH. (The SDK's wait_stop()
        only waits for 'not running', so it returns instantly if called before
        the controller has switched to PLAY - a race the ROS driver hits.)"""
        t0 = time.time()
        while self.state() != 3 and time.time() - t0 < start_timeout:
            time.sleep(0.005)
        while True:
            s = self.state()
            if s != 3:
                if s != 0:
                    raise RpcError(f"motion ended in state {ROBOT_STATE.get(s, s)} alarm={self.alarm()}")
                return
            if time.time() - t0 > timeout:
                self.stop()
                raise RpcError("motion timeout - stopped")
            time.sleep(0.01)

    def movej(self, j, speed=10, acc=None, dec=None, wait=True):
        """Joint-space PTP. j in degrees. speed in % (1..100)."""
        p = {"targetPos": pad8(j), "speed": speed}
        if acc is not None: p["acc"] = acc
        if dec is not None: p["dec"] = dec
        ok = self.call("moveByJoint", p)
        if ok is not True:
            raise RpcError(f"moveByJoint rejected: {ok}")
        if wait: self.wait_motion()
        return ok

    def movel(self, j, speed=100, speed_type=0, acc=None, dec=None, wait=True):
        """Straight-line TCP motion. NOTE target is a JOINT vector (deg) - get it
        from ik() first. speed_type 0: mm/s (1..3000), 1: deg/s rotational."""
        p = {"targetPos": pad8(j), "speed": speed, "speed_type": speed_type}
        if acc is not None: p["acc"] = acc
        if dec is not None: p["dec"] = dec
        ok = self.call("moveByLine", p)
        if ok is not True:
            raise RpcError(f"moveByLine rejected: {ok}")
        if wait: self.wait_motion()
        return ok

    def jog(self, index, speed=None):
        """index 0..11: axis=index//2, even=negative, odd=positive. Must be resent
        < 1 s apart or the controller stops. Call stop() to end."""
        return self.call("jog", {"index": index, **({"speed": speed} if speed else {})})

    def speedj(self, vj_deg_s, acc=50, t=0.1):
        return self.call("moveBySpeedj", {"vj": pad8(vj_deg_s), "acc": acc, "t": t})

    def stopj(self, acc=100):
        return self.call("stopj", {"acc": acc})

    # ---- time-stamped trajectory ("ML" / moveml) -----------------------------
    def run_trajectory(self, times_s, joints_deg, speed_percent=100.0, wait=True):
        """Upload & execute a time-stamped joint trajectory.
        times_s[0] must be 0 and strictly increasing; joints_deg[0] should be the
        current position. This is what the ROS controller uses (joint_trajectory_movel)."""
        cur = self.joints()
        self.call("start_push_pos", {"path_lenth": len(times_s), "pos_type": 0,
                                     "ref_joint_pos": cur, "ref_frame": [0, 0, 0, 0, 0, 0],
                                     "ret_flag": 1})
        for t, q in zip(times_s, joints_deg):
            # ret_flag=1 -> controller acks every point, keeps the socket in lock-step
            self.call("push_pos", {"timestamp": round(float(t), 6), "pos": list(q)[:6]})
        self.call("stop_push_pos")
        r = self.call("check_trajectory")
        if r != 0:
            self.call("flush_trajectory")
            raise RpcError(f"trajectory rejected: {ML_PUSH_RESULT.get(r, r)}")
        self.call("start_trajectory", {"speed_percent": speed_percent})
        if wait: self.wait_motion()

    # ---- I/O ------------------------------------------------------------------
    def set_do(self, addr, value):  return self.call("setOutput", {"addr": addr, "status": int(value)})
    def get_di(self, addr):         return self.call("getInput", {"addr": addr})
    def get_do(self, addr):         return self.call("getOutput", {"addr": addr})
    def set_ao(self, addr, value):  return self.call("setAnalogOutput", {"addr": addr, "value": value})
    def get_ai(self, addr):         return self.call("getAnalogInput", {"addr": addr})


def info(rpc):
    """Everything the `info` command prints, keyed by name; failed queries hold the error text."""
    out = {}
    for k, f in [("version", rpc.version), ("subtype", rpc.subtype), ("mode", rpc.mode), ("state", rpc.state),
                 ("servo", rpc.servo_status), ("synced", rpc.sync_status), ("estop", rpc.estop), ("speed", rpc.speed),
                 ("alarm", rpc.alarm), ("joints", rpc.joints), ("tcp_pose", rpc.tcp_pose), ("DH", rpc.dh)]:
        try:
            out[k] = f()
        except Exception as e:  # noqa
            out[k] = f"<{e}>"
    return out


def pad8(j):
    j = [float(v) for v in j]
    return (j + [0.0] * 8)[:8]


# --------------------------------------------------------------------------- 8056
# Packet layout (big-endian, no padding). Fields are appended over firmware
# versions; MessageSize (first u32) is the total packet length incl. itself.
# Parse only as many fields as the packet actually contains.
STATE_FIELDS = [
    ("MessageSize", "I"),            # bytes in this packet
    ("TimeStamp", "Q"),              # ms since epoch
    ("autorun_cycleMode", "B"),      # 0 step, 1 cycle, 2 continuous
    ("machinePos", "8d"),            # joint angles, deg
    ("machinePose", "6d"),           # TCP pose in base: mm, rad
    ("machineUserPose", "6d"),       # TCP pose in user frame
    ("torque", "8d"),                # joint torque (per-mille of rated)
    ("robotState", "i"),             # 0 stop 1 pause 2 estop 3 running 4 error 5 collision
    ("servoReady", "i"),
    ("can_motor_run", "i"),
    ("motor_speed", "8i"),           # rpm
    ("robotMode", "i"),              # 0 teach 1 play 2 remote
    ("analog_ioInput", "3d"),
    ("analog_ioOutput", "5d"),
    ("digital_ioInput", "Q"),        # bitmask X0..X63
    ("digital_ioOutput", "Q"),       # bitmask Y0..Y63
    ("collision", "B"),
    ("machineFlangePose", "6d"),     # flange pose in base: mm, rad
    ("machineUserFlangePose", "6d"),
    ("emergencyStopState", "B"),
    ("tcp_speed", "d"),
    ("joint_speed", "8d"),           # deg/s
    ("tcpacc", "d"),
    ("jointacc", "8d"),
]


def parse_state(pkt):
    out, off = {}, 0
    for name, fmt in STATE_FIELDS:
        f = "!" + fmt
        n = struct.calcsize(f)
        if off + n > len(pkt):
            break
        v = struct.unpack_from(f, pkt, off)
        out[name] = list(v) if len(v) > 1 else v[0]
        off += n
    return out


class StateStream:
    """Background reader for port 8056. `latest` holds the newest decoded packet."""

    def __init__(self, ip, port=STATE_PORT):
        self.sock = socket.create_connection((ip, port), timeout=5.0)
        self.latest = None
        self.count = 0
        self._run = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _recv_exact(self, n):
        b = b""
        while len(b) < n:
            c = self.sock.recv(n - len(b))
            if not c:
                raise ConnectionError("8056 closed")
            b += c
        return b

    def _loop(self):
        while self._run:
            head = self._recv_exact(4)
            size = struct.unpack("!I", head)[0]
            if size < 4 or size > 65536:
                raise ConnectionError(f"lost sync on 8056 (size={size})")
            self.latest = parse_state(head + self._recv_exact(size - 4))
            self.count += 1

    def wait_first(self, timeout=3.0):
        t0 = time.time()
        while self.latest is None:
            if time.time() - t0 > timeout:
                raise TimeoutError("no packet on 8056")
            time.sleep(0.01)
        return self.latest

    def close(self):
        self._run = False
        self.sock.close()


# --------------------------------------------------------------------------- kinematics
# Official standard-DH (EC66 manual Table 3-2; EC63/EC612 same structure).
# owl_description URDFs reproduce this exactly (checked numerically).
DH = {  # d1, a2, a3, d4, d5, d6   [m]
    "owl63": (0.140, 0.270, 0.256, 0.103, 0.098, 0.089),
    "owl66": (0.096, 0.418, 0.398, 0.122, 0.098, 0.089),
    "owl612": (0.185, 0.615, 0.572, 0.174, 0.116, 0.103),
}


def dh_fk(q_deg, robot):
    d1, a2, a3, d4, d5, d6 = DH[robot]
    rows = [(0, -math.pi / 2, d1), (a2, 0, 0), (a3, 0, 0), (0, -math.pi / 2, d4), (0, -math.pi / 2, d5), (0, 0, d6)]
    T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for (a, al, d), qd in zip(rows, q_deg[:6]):
        t = math.radians(qd)
        ct, st, ca, sa = math.cos(t), math.sin(t), math.cos(al), math.sin(al)
        A = [[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]]
        T = [[sum(T[i][k] * A[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    return T


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.200")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("monitor")
    sub.add_parser("info")
    sub.add_parser("servo_on")
    sub.add_parser("servo_off")
    sub.add_parser("stop")
    mj = sub.add_parser("movej"); mj.add_argument("j", type=float, nargs=6); mj.add_argument("--speed", type=float, default=10)
    ck = sub.add_parser("checkfk"); ck.add_argument("--robot", choices=list(DH), required=True)
    sub.add_parser("io")
    do = sub.add_parser("do"); do.add_argument("addr", type=int); do.add_argument("value", type=int, choices=[0, 1])
    pu = sub.add_parser("pulse"); pu.add_argument("addr", type=int); pu.add_argument("seconds", type=float, nargs="?", default=1.0)
    rw = sub.add_parser("raw"); rw.add_argument("method"); rw.add_argument("params", nargs="?", default="{}")
    a = ap.parse_args()

    if a.cmd == "monitor":
        s = StateStream(a.ip); s.wait_first()
        try:
            while True:
                d = s.latest
                print(f"[{s.count:6d}] size={d['MessageSize']} mode={ROBOT_MODE.get(d.get('robotMode'))} "
                      f"state={ROBOT_STATE.get(d.get('robotState'))} servo={d.get('servoReady')} "
                      f"q(deg)={[round(x, 2) for x in d['machinePos'][:6]]} "
                      f"flange(mm,rad)={[round(x, 3) for x in d.get('machineFlangePose', [])]}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            s.close()
        return

    if a.cmd == "io":
        # Read-only snapshot from 8056. Y48/Y49 + X48/X49 = wrist tool connector,
        # Y0-Y19 / X4-X19 = control box, X50/X51 = flange buttons.
        s = StateStream(a.ip); d = s.wait_first(); s.close()
        on = lambda mask: [i for i in range(64) if mask >> i & 1]
        print("digital outputs ON (Y):", on(d["digital_ioOutput"]))
        print("digital inputs  ON (X):", on(d["digital_ioInput"]))
        print("tool  Y48,Y49 / X48,X49:", [d["digital_ioOutput"] >> i & 1 for i in (48, 49)],
              [d["digital_ioInput"] >> i & 1 for i in (48, 49)])
        print("analog in  (AI1..3)    :", [round(v, 3) for v in d["analog_ioInput"]])
        print("analog out (AO1..5)    :", [round(v, 3) for v in d["analog_ioOutput"]])
        return

    rpc = EliteRPC(a.ip)
    try:
        if a.cmd == "do":
            print(f"Y{a.addr} <- {a.value}:", rpc.set_do(a.addr, a.value), "| readback:", rpc.get_do(a.addr))
            return
        if a.cmd == "pulse":
            print(f"Y{a.addr} ON for {a.seconds}s:", rpc.set_do(a.addr, 1))
            try:
                time.sleep(a.seconds)
            finally:
                print(f"Y{a.addr} OFF:", rpc.set_do(a.addr, 0))
            return
        if a.cmd == "info":
            for k, v in info(rpc).items():
                print(f"{k:9s}: {v}")
        elif a.cmd == "servo_on":
            print(rpc.servo_on())
        elif a.cmd == "servo_off":
            print(rpc.servo_off())
        elif a.cmd == "stop":
            print(rpc.stop())
        elif a.cmd == "movej":
            rpc.movej(a.j, speed=a.speed); print("done", rpc.joints())
        elif a.cmd == "raw":
            print(json.dumps(rpc.call(a.method, json.loads(a.params))))
        elif a.cmd == "checkfk":
            q = rpc.joints()
            ctrl = rpc.fk(q)                  # controller FK (mm, rad) incl. active tool
            fl = rpc.flange_pose()            # flange only (mm, rad)
            T = dh_fk(q, a.robot)
            ours = [T[0][3] * 1000, T[1][3] * 1000, T[2][3] * 1000]
            print("joints (deg)            :", [round(x, 3) for x in q[:6]])
            print("controller flange xyz mm:", [round(x, 2) for x in fl[:3]])
            print("URDF/DH   flange xyz mm:", [round(x, 2) for x in ours])
            print("controller FK (w/ tool) :", [round(x, 3) for x in ctrl])
            err = math.dist(fl[:3], ours)
            print(f"flange position error   : {err:.2f} mm  ->", "OK" if err < 2 else "MISMATCH: zero/DH/robot type differ")
            # getDH link list (mm): [d1, shoulder_y, a2, elbow_y, a3, wrist1_y, d5, d6, ...]
            # net d4 = shoulder_y - elbow_y + wrist1_y. Verified on an EC63 (fw V3.11.2).
            D = rpc.dh()
            cal = (D[0], D[2], D[4], D[1] - D[3] + D[5], D[6], D[7])
            DH["calibrated"] = tuple(v / 1000 for v in cal)
            Tc = dh_fk(q, "calibrated")
            ours_c = [Tc[0][3] * 1000, Tc[1][3] * 1000, Tc[2][3] * 1000]
            print("calibrated d1,a2,a3,d4,d5,d6 mm:", [round(v, 3) for v in cal])
            print(f"calibrated-DH flange err : {math.dist(fl[:3], ours_c):.2f} mm")
    finally:
        rpc.close()


if __name__ == "__main__":
    main()
