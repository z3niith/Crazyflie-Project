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
trajectory_x = []
trajectory_y = []
range_distances = {"Front": 0, "Back": 0, "Left": 0, "Right": 0, "Up": 0}
recording = False

AVOID_THRESHOLD = 0.5  # meters
AVOID_STEP = 0.1  # max avoidance correction in meters

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, deadzone=0.02, output_limit=0.2):
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
    global current_x, current_y, current_z, trajectory_x, trajectory_y, range_distances
    current_x = data.get('stateEstimate.x', current_x)
    current_y = data.get('stateEstimate.y', current_y)
    current_z = data.get('stateEstimate.z', current_z)

    trajectory_x.append(current_x)
    trajectory_y.append(current_y)

    range_distances["Front"] = data.get('range.front', 0) / 1000.0
    range_distances["Back"] = data.get('range.back', 0) / 1000.0
    range_distances["Left"] = data.get('range.left', 0) / 1000.0
    range_distances["Right"] = data.get('range.right', 0) / 1000.0
    range_distances["Up"] = data.get('range.up', 0) / 1000.0

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def hover_with_obstacle_avoidance(commander, pid_x, pid_y, setpoint_x, setpoint_y, altitude, duration, dt=0.05):
    start = time.time()
    last_time = start
    while time.time() - start < duration:
        now = time.time()
        elapsed = now - last_time
        last_time = now

        avoid_dx = 0.0
        avoid_dy = 0.0

        if range_distances["Front"] < AVOID_THRESHOLD:
            avoid_dy -= AVOID_STEP
        if range_distances["Back"] < AVOID_THRESHOLD:
            avoid_dy += AVOID_STEP
        if range_distances["Left"] < AVOID_THRESHOLD:
            avoid_dx += AVOID_STEP
        if range_distances["Right"] < AVOID_THRESHOLD:
            avoid_dx -= AVOID_STEP

        corr_x = pid_x.update(current_x, elapsed)
        corr_y = pid_y.update(current_y, elapsed)

        new_x = setpoint_x + corr_x + avoid_dx
        new_y = setpoint_y + corr_y + avoid_dy

        commander.go_to(new_x, new_y, altitude, yaw=0.0, duration_s=0.2, relative=False)
        time.sleep(dt)

def reset_estimator(scf):
    print("[INFO] Resetting Kalman estimator...")
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(1.0)

def drone_flight():
    global recording
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()

    URI = 'radio://0/80/2M'
    target_altitude = 1.0
    hover_duration = 10.0

    pid_x = PIDController(0.05, 0.002, 0.02)
    pid_y = PIDController(0.05, 0.002, 0.02)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        reset_estimator(scf)

        log_conf = LogConfig(name='State', period_in_ms=50)
        for var in ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                    'range.front', 'range.back', 'range.left', 'range.right', 'range.up']:
            log_conf.add_variable(var, 'float' if 'stateEstimate' in var else 'uint16_t')

        threading.Thread(target=log_thread_func, args=(scf, log_conf), daemon=True).start()

        time.sleep(2.0)
        commander = HighLevelCommander(scf.cf)
        recording = True

        commander.takeoff(target_altitude, 1.0)
        time.sleep(2.0)

        pid_x.setpoint = current_x
        pid_y.setpoint = current_y

        hover_with_obstacle_avoidance(commander, pid_x, pid_y, current_x, current_y, target_altitude, hover_duration)

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

range_bars = ax_range.bar(range(5), [0]*5, tick_label=["Front", "Back", "Left", "Right", "Up"])
ax_range.set_ylim(0, 2)
ax_range.set_title("Multiranger (m)")
ax_range.set_ylabel("Distance (m)")

def update(frame):
    traj_line.set_data(trajectory_x, trajectory_y)
    for i, dir in enumerate(["Front", "Back", "Left", "Right", "Up"]):
        range_bars[i].set_height(range_distances[dir])
    return traj_line, *range_bars

threading.Thread(target=drone_flight, daemon=True).start()
ani = FuncAnimation(fig, update, interval=100)
plt.tight_layout()
plt.show()