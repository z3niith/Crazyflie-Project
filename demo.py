import time
import threading
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig

# Global state
current_x = 0.0
current_y = 0.0
current_z = 0.0
current_roll = 0.0
current_pitch = 0.0
current_yaw = 0.0
trajectory_x = []
trajectory_y = []
range_distances = {"Front": 0, "Back": 0, "Left": 0, "Right": 0, "Up": 0, "Down": 0}
recording = False

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, deadzone=0.02, output_limit=0.03):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_error = 0.0
        self.deadzone = deadzone
        self.output_limit = output_limit

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        if abs(error) < self.deadzone:
            return 0.0
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return max(min(output, self.output_limit), -self.output_limit)

def log_callback(timestamp, data, logconf):
    global current_x, current_y, current_z, trajectory_x, trajectory_y
    global current_roll, current_pitch, current_yaw, range_distances

    current_x = data.get('stateEstimate.x', current_x)
    current_y = data.get('stateEstimate.y', current_y)
    current_z = data.get('stateEstimate.z', current_z)
    current_roll = data.get('stabilizer.roll', current_roll)
    current_pitch = data.get('stabilizer.pitch', current_pitch)
    current_yaw = data.get('stabilizer.yaw', current_yaw)

    trajectory_x.append(current_x)
    trajectory_y.append(current_y)

    range_distances["Front"] = data.get('range.front', 0) / 1000.0
    range_distances["Back"] = data.get('range.back', 0) / 1000.0
    range_distances["Left"] = data.get('range.left', 0) / 1000.0
    range_distances["Right"] = data.get('range.right', 0) / 1000.0
    range_distances["Up"] = data.get('range.up', 0) / 1000.0
    range_distances["Down"] = data.get('range.zrange', 0) / 1000.0

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def hover(commander, pid_x, pid_y, pid_z, setpoint_x, setpoint_y, base_altitude, duration, dt=0.05):
    start = time.time()
    last_time = start
    while time.time() - start < duration:
        now = time.time()
        elapsed = now - last_time
        last_time = now

        corr_x = pid_x.update(current_x, elapsed)
        corr_y = pid_y.update(current_y, elapsed)

        # Smooth altitude adjustment if down sensor detects proximity
        down_range = range_distances.get("Down", 1.0)
        if down_range < 0.3:
            pid_z.setpoint = base_altitude + 0.3
        else:
            pid_z.setpoint = base_altitude
        new_z = base_altitude + pid_z.update(current_z, elapsed)

        commander.go_to(setpoint_x + corr_x, setpoint_y + corr_y, new_z, 0.0, 0.2, False)
        time.sleep(dt)

def reset_estimator(scf):
    print("[INFO] Resetting Kalman estimator...")
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(1.0)

def wait_for_stable_orientation(timeout=10):
    print("[INFO] Checking for stable roll/pitch before takeoff...")
    start = time.time()
    while time.time() - start < timeout:
        if abs(current_roll) < 1.5 and abs(current_pitch) < 1.5:
            print(f"Stable: Roll={current_roll:.2f}, Pitch={current_pitch:.2f}")
            return True
        print(f"Waiting... Roll={current_roll:.2f}, Pitch={current_pitch:.2f}")
        time.sleep(0.5)
    print("[WARN] Roll/Pitch never stabilized.")
    return False

def drone_flight():
    global recording
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()

    URI = 'radio://0/80/2M'
    target_altitude = 1.0
    hover_duration = 10.0

    pid_x = PIDController(0.05, 0.002, 0.02)
    pid_y = PIDController(0.05, 0.002, 0.02)
    pid_z = PIDController(0.3, 0.0, 0.1, setpoint=target_altitude, output_limit=0.3)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        log_conf = LogConfig(name='State', period_in_ms=50)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stabilizer.roll', 'float')
        log_conf.add_variable('stabilizer.pitch', 'float')
        log_conf.add_variable('stabilizer.yaw', 'float')
        log_conf.add_variable('range.front', 'uint16_t')
        log_conf.add_variable('range.back', 'uint16_t')
        log_conf.add_variable('range.left', 'uint16_t')
        log_conf.add_variable('range.right', 'uint16_t')
        log_conf.add_variable('range.up', 'uint16_t')
        log_conf.add_variable('range.zrange', 'uint16_t')

        threading.Thread(target=log_thread_func, args=(scf, log_conf), daemon=True).start()
        time.sleep(2.0)

        reset_estimator(scf)
        time.sleep(2.0)

        if not wait_for_stable_orientation():
            reset_estimator(scf)
            time.sleep(2.0)
            wait_for_stable_orientation()

        commander = HighLevelCommander(scf.cf)
        recording = True

        commander.takeoff(target_altitude, 1.0)
        time.sleep(2.0)

        pid_x.setpoint = current_x
        pid_y.setpoint = current_y

        hover(commander, pid_x, pid_y, pid_z, current_x, current_y, target_altitude, hover_duration)

        commander.land(0.0, 2.0)
        time.sleep(3.0)
        recording = False

# === Visualization ===
fig, (ax_traj, ax_range) = plt.subplots(1, 2, figsize=(12, 6))

traj_line, = ax_traj.plot([], [], lw=2)
ax_traj.set_xlim(-2, 2)
ax_traj.set_ylim(-2, 2)
ax_traj.set_title("XY Trajectory")
ax_traj.set_xlabel("X (m)")
ax_traj.set_ylabel("Y (m)")
ax_traj.grid(True)

range_bars = ax_range.bar(range(6), [0]*6, tick_label=["Front", "Back", "Left", "Right", "Up", "Down"])
ax_range.set_ylim(0, 2)
ax_range.set_title("Multiranger (m)")
ax_range.set_ylabel("Distance (m)")

def update(frame):
    traj_line.set_data(trajectory_x, trajectory_y)
    for i, dir in enumerate(["Front", "Back", "Left", "Right", "Up", "Down"]):
        range_bars[i].set_height(range_distances[dir])
    return traj_line, *range_bars

threading.Thread(target=drone_flight, daemon=True).start()
ani = FuncAnimation(fig, update, interval=100)
plt.tight_layout()
plt.show()
