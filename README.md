# OWL 6.3 Control
The OWL 6.3 is an Elite Robots EC63. This repo talks to its controller directly
and plans collision-free motion with PyRoki.

## Safety and Tips
- The **E-STOP is on the pendant** and not on the table.
- Setting the pendant into teach or play mode turn's off the servo control mode.
  Use the gui button or direct command to turn servo on again, **after** putting
  pendant back to remote mode.
- You can use the teach mode's free move option to move the robot freely to any
  position just by moving it yourself. These positions are updated in the
  control firmware as well and when used with the positions tab allow you to
  easily save position points.

**Dodgy things**:
- The collisions are calculated by a buffer around the links. This can be pretty
  dodgy and many reachable positions show up as unreachable. I think of it as a
  safety feature but feel free to mess about with that (not suggested for legal
  reasons).
- The sequence execution is also a bit dodgy and buggy when:
    - Connection to the robot or control server drops.
    - You reset the scene.
    - The starting point in sequence is the point the robot is already at.


## How communication works
Your computer can connect to the controller directly on the robot's subnet via
the Ethernet port. The robot controller is at `192.168.1.200`. Set your device's
ip to `192.168.1.100` and subnet mask as `255.255.255.0` on the Ethernet port.
Communication to the robot occurs over two TCP ports:

| Port | Direction | Format | Use |
|---|---|---|---|
| 8055 | Device -> robot | JSON-RPC, one JSON object per line | Commands and queries |
| 8056 | Robot -> device | Binary packet every 8 ms | Live state: joints, pose, mode, I/O |

A command looks like this:

```
-> {"method":"getRobotState","params":{},"jsonrpc":"2.0","id":1}
<- {"jsonrpc":"2.0","result":"0","id":1}
```

Joints are in degrees, positions in mm, rotations in radians. Motion, outputs and `setSpeed` need the pendant key on **REMOTE**.

A planned move is sent as a time-stamped trajectory: `start_push_pos`, one `push_pos` per 8 ms point, `stop_push_pos`, `check_trajectory`, `start_trajectory`.

## Commands

```bash
python owl_direct.py --ip 192.168.1.200 <command>
```

| Command | What you see or what it does |
|---|---|
| `info` | Version, model, mode, state, servo, speed, alarms, joints, pose, DH |
| `monitor` | Live state from port 8056 |
| `io` | Digital and analog inputs and outputs |
| `checkfk --robot owl63` | Controller flange position vs the model |
| `servo_on` / `servo_off` | Enable or disable the servos |
| `movej J1 .. J6 --speed 10` | Joint move in degrees, speed in % |
| `do 1 1` / `do 1 0` | Suction on / off (output Y1) |
| `stop` | Stop motion |
| `raw METHOD '{json}'` | Any controller command, for example `raw getSpeed` or `raw setSpeed '{"value": 100}'` |

Useful raw methods: `getRobotMode` (0 teach, 1 play, 2 remote), `getRobotState` (0 stop, 3 running, 4 error), `get_joint_pos`, `getServoStatus`, `getSpeed`, `setSpeed`, `setOutput`.


## Using the repo
Setup once:

```bash
conda env create -f environment.yml
conda activate arm
pip install "git+https://github.com/chungmin99/pyroki.git"
```

Run the GUI (Leave out `--ip` to simulate):

```bash
python plan_gui.py --ip 192.168.1.200
```

1. Put the table and wall in `scene.toml` (metres, robot base frame), then press
   **Reload scene**.
    - Use the rendering engine's setting menu to toggle on world coordinate axes
      and item labels to make it easier to 
2. **Motion** tab: drag the target, press **Plan**, check the preview, press **Execute**.
3. **Positions** tab: save poses by name, go **Home**, and run sequences such as `pick, SUCK, wait(300), place, UNSUCK` for a number of cycles or forever.
4. **Robot** tab: live data, suction, controller speed, info and raw commands.


| File | Purpose |
|---|---|
| `owl_direct.py` | Controller client and CLI |
| `planner.py` | Robot model, IK, path planning, execution |
| `plan_gui.py`, `robot_panel.py` | Browser GUI |
| `scene.toml` | Tool size and obstacle boxes |
| `positions.json` | Saved positions and sequence |
| `owl_description/` | Arm meshes for display |

Start at low speed with a hand on the e-stop.




## References

```
@inproceedings{kim2025pyroki,
  title={PyRoki: A Modular Toolkit for Robot Kinematic Optimization},
  author={Kim*, Chung Min and Yi*, Brent and Choi, Hongsuk and Ma, Yi and Goldberg, Ken and Kanazawa, Angjoo},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  url={https://arxiv.org/abs/2505.03728},
}
```