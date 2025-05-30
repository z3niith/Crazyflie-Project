import time
import threading
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.high_level_commander import HighLevelCommander

# Global state variables
altitude = 0.0
velocity_z = 0.0
thrust = 0
roll = 0.0
pitch = 0.0
yaw = 0.0
m1 = m2 = m3 = m4 = 0

trajectory_x, trajectory_y, trajectory_z = [], [], []
alt_data, vz_data, thrust_data = [], [], []
roll_data, pitch_data, yaw_data = [], [], []
m1_data, m2_data, m3_data, m4_data = [], [], [], []
range_distances = {"Front": 0, "Back": 0, "Left": 0, "Right": 0, "Up": 0, "Down": 0}

WINDOW = 200
plot_running = True
URI = 'radio://0/80/2M'
THRUST_MAX = 50000
ROLL_PITCH_THRESHOLD = 30.0

def log_callback(timestamp, data, logconf):
    global altitude, velocity_z, thrust, roll, pitch, yaw, m1, m2, m3, m4
    altitude = data.get('stateEstimate.z', altitude)
    velocity_z = data.get('stateEstimate.vz', velocity_z)
    thrust = data.get('stabilizer.thrust', thrust)
    roll = data.get('stabilizer.roll', roll)
    pitch = data.get('stabilizer.pitch', pitch)
    yaw = data.get('stabilizer.yaw', yaw)
    m1 = data.get('pwm.m1', m1)
    m2 = data.get('pwm.m2', m2)
    m3 = data.get('pwm.m3', m3)
    m4 = data.get('pwm.m4', m4)

    trajectory_x.append(data.get('stateEstimate.x', 0))
    trajectory_y.append(data.get('stateEstimate.y', 0))
    trajectory_z.append(altitude)

    alt_data.append(altitude)
    vz_data.append(velocity_z)
    thrust_data.append(thrust)
    roll_data.append(roll)
    pitch_data.append(pitch)
    yaw_data.append(yaw)
    m1_data.append(m1)
    m2_data.append(m2)
    m3_data.append(m3)
    m4_data.append(m4)

    range_distances["Front"] = data.get('range.front', 0) / 1000.0
    range_distances["Back"] = data.get('range.back', 0) / 1000.0
    range_distances["Left"] = data.get('range.left', 0) / 1000.0
    range_distances["Right"] = data.get('range.right', 0) / 1000.0
    range_distances["Up"] = data.get('range.up', 0) / 1000.0
    range_distances["Down"] = data.get('range.zrange', 0) / 1000.0

    if len(alt_data) > WINDOW:
        alt_data.pop(0); vz_data.pop(0); thrust_data.pop(0)
        roll_data.pop(0); pitch_data.pop(0); yaw_data.pop(0)
        m1_data.pop(0); m2_data.pop(0); m3_data.pop(0); m4_data.pop(0)
        trajectory_x.pop(0); trajectory_y.pop(0); trajectory_z.pop(0)

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            if not plot_running:
                break
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def plot_live():
    fig = plt.figure(figsize=(15, 10))
    plot_pairs = [
        ("Altitude (z)", alt_data),
        ("Vertical Velocity (vz)", vz_data),
        ("Thrust", thrust_data),
        ("Roll", roll_data),
        ("Pitch", pitch_data),
        ("Yaw", yaw_data),
        ("Motor 1", m1_data),
        ("Motor 2", m2_data),
        ("Motor 3", m3_data),
        ("Motor 4", m4_data)
    ]

    rows_needed = len(plot_pairs) + 3
    gs = fig.add_gridspec(rows_needed, 1)

    ax3d = fig.add_subplot(gs[0:2, 0], projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.set_xlim([-1, 1])
    ax3d.set_ylim([-1, 1])
    ax3d.set_zlim([0, 2])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    traj_line, = ax3d.plot([], [], [], lw=2)

    ax_mr = fig.add_subplot(gs[2, 0])
    ax_mr.set_title("Multiranger Readings")
    bar_labels = ["Front", "Back", "Left", "Right", "Up", "Down"]
    bars = ax_mr.bar(bar_labels, [0]*6)
    ax_mr.set_ylim(0, 2)

    axes = [fig.add_subplot(gs[i + 3, 0]) for i in range(len(plot_pairs))]
    lines = []
    for ax, (title, data) in zip(axes, plot_pairs):
        ax.set_title(title)
        ax.set_xlim(0, WINDOW)
        ax.grid(True)
        line, = ax.plot([], [], label=title)
        if title == "Thrust":
            ax.axhline(THRUST_MAX, color='r', linestyle='--', label='Thrust Max')
        elif title in ["Roll", "Pitch"]:
            ax.axhline(ROLL_PITCH_THRESHOLD, color='orange', linestyle='--')
            ax.axhline(-ROLL_PITCH_THRESHOLD, color='orange', linestyle='--')
        ax.legend()
        lines.append(line)

    def update(frame):
        if trajectory_x and trajectory_y and trajectory_z:
            traj_line.set_data(trajectory_x, trajectory_y)
            traj_line.set_3d_properties(trajectory_z)
        for i, label in enumerate(bar_labels):
            bars[i].set_height(range_distances[label])
        for line, (_, data) in zip(lines, plot_pairs):
            line.set_data(range(len(data)), data)
            line.axes.relim()
            line.axes.autoscale_view()
        return lines

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

def main():
    global plot_running
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        log_conf = LogConfig(name='FullDiagnostics', period_in_ms=50)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stateEstimate.vz', 'float')
        log_conf.add_variable('stabilizer.thrust', 'uint16_t')
        log_conf.add_variable('stabilizer.roll', 'float')
        log_conf.add_variable('stabilizer.pitch', 'float')
        log_conf.add_variable('stabilizer.yaw', 'float')
        log_conf.add_variable('pwm.m1', 'uint16_t')
        log_conf.add_variable('pwm.m2', 'uint16_t')
        log_conf.add_variable('pwm.m3', 'uint16_t')
        log_conf.add_variable('pwm.m4', 'uint16_t')
        log_conf.add_variable('range.front', 'uint16_t')
        log_conf.add_variable('range.back', 'uint16_t')
        log_conf.add_variable('range.left', 'uint16_t')
        log_conf.add_variable('range.right', 'uint16_t')
        log_conf.add_variable('range.up', 'uint16_t')
        log_conf.add_variable('range.zrange', 'uint16_t')

        log_thread = threading.Thread(target=log_thread_func, args=(scf, log_conf), daemon=True)
        log_thread.start()

        commander = HighLevelCommander(scf.cf)
        commander.takeoff(0.5, 2.0)
        time.sleep(3.0)
        commander.hover(0.0, 0.0, 0.5, 0.0, 5.0)
        time.sleep(5.0)
        commander.land(0.0, 2.0)
        time.sleep(3.0)

        plot_running = False

if __name__ == '__main__':
    t = threading.Thread(target=main)
    t.start()
    plot_live()
