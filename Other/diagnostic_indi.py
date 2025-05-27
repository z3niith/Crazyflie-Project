import time
import threading
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.high_level_commander import HighLevelCommander

# Global variables for telemetry
altitude = 0.0
velocity_z = 0.0
thrust = 0
roll = 0.0
pitch = 0.0
yaw = 0.0
motor_m1 = 0
motor_m2 = 0
motor_m3 = 0
motor_m4 = 0

# Data lists for plotting
WINDOW = 200
alt_data, vz_data, thrust_data = [], [], []
roll_data, pitch_data, yaw_data = [], [], []
m1_data, m2_data, m3_data, m4_data = [], [], [], []

plot_running = True

URI = 'radio://0/80/2M'
THRUST_MAX = 50000
ROLL_PITCH_THRESHOLD = 30.0

# Logging callback
def log_callback(timestamp, data, logconf):
    global altitude, velocity_z, thrust, roll, pitch, yaw
    global motor_m1, motor_m2, motor_m3, motor_m4

    altitude = data.get('stateEstimate.z', altitude)
    velocity_z = data.get('stateEstimate.vz', velocity_z)
    thrust = data.get('stabilizer.thrust', thrust)
    roll = data.get('stabilizer.roll', roll)
    pitch = data.get('stabilizer.pitch', pitch)
    yaw = data.get('stabilizer.yaw', yaw)
    motor_m1 = data.get('pwm.m1', motor_m1)
    motor_m2 = data.get('pwm.m2', motor_m2)
    motor_m3 = data.get('pwm.m3', motor_m3)
    motor_m4 = data.get('pwm.m4', motor_m4)

    for buf, val in zip(
        [alt_data, vz_data, thrust_data, roll_data, pitch_data, yaw_data, m1_data, m2_data, m3_data, m4_data],
        [altitude, velocity_z, thrust, roll, pitch, yaw, motor_m1, motor_m2, motor_m3, motor_m4]
    ):
        buf.append(val)
        if len(buf) > WINDOW:
            buf.pop(0)

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            if not plot_running:
                break
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def reset_kalman(scf):
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(0.5)

def flight_test_sequence(commander):
    commander.takeoff(0.5, 2.0)
    time.sleep(3.0)
    time.sleep(2.0)
    commander.go_to(0.0, 0.0, 1.3, 0.0, 2.0, False)
    time.sleep(2.5)
    commander.go_to(0.0, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)
    commander.go_to(0.5, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)
    commander.go_to(0.0, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)
    commander.go_to(0.0, 0.5, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)
    commander.go_to(0.0, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)
    commander.go_to(0.0, 0.0, 1.0, 180.0, 3.0, False)
    time.sleep(3.0)
    commander.go_to(0.0, 0.0, 1.0, 0.0, 3.0, False)
    time.sleep(3.0)
    commander.land(0.0, 2.0)
    time.sleep(3.0)

def run_diagnostics():
    global plot_running
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        log_conf = LogConfig(name='FullDiagnostics', period_in_ms=100)
        variables = [
            ('stateEstimate.z', 'float'),
            ('stateEstimate.vz', 'float'),
            ('stabilizer.thrust', 'uint16_t'),
            ('stabilizer.roll', 'float'),
            ('stabilizer.pitch', 'float'),
            ('stabilizer.yaw', 'float'),
            ('pwm.m1', 'uint16_t'),
            ('pwm.m2', 'uint16_t'),
            ('pwm.m3', 'uint16_t'),
            ('pwm.m4', 'uint16_t')
        ]

        for name, dtype in variables:
            try:
                log_conf.add_variable(name, dtype)
            except KeyError:
                print(f"[WARNING] Variable '{name}' not found on this firmware build.")

        if not log_conf.variables:
            print("[ERROR] No valid log variables found. Exiting.")
            return

        threading.Thread(target=log_thread_func, args=(scf, log_conf), daemon=True).start()

        commander = HighLevelCommander(scf.cf)
        reset_kalman(scf)
        flight_test_sequence(commander)
        plot_running = False

def plot_live():
    fig, axs = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle("Crazyflie Live Telemetry")

    lines = []
    titles = [
        "Altitude (z)", "Vertical Velocity (vz)", "Thrust",
        "Roll", "Pitch", "Yaw",
        "Motors 1-2", "Motors 3-4"
    ]

    data_sources = [
        alt_data, vz_data, thrust_data,
        roll_data, pitch_data, yaw_data,
        (m1_data, m2_data), (m3_data, m4_data)
    ]

    for ax, title, data in zip(axs.flat, titles, data_sources):
        ax.set_title(title)
        ax.set_xlim(0, WINDOW)
        ax.grid(True)
        if isinstance(data, tuple):
            l1, = ax.plot([], [], label='Motor1')
            l2, = ax.plot([], [], label='Motor2' if title.endswith("1-2") else 'Motor3')
            lines.extend([l1, l2])
            ax.legend()
        else:
            line, = ax.plot([], [], label=title)
            lines.append(line)
            if title == "Thrust":
                ax.axhline(THRUST_MAX, color='red', linestyle='--', label='Thrust Max')
            elif title in ["Roll", "Pitch"]:
                ax.axhline(ROLL_PITCH_THRESHOLD, color='orange', linestyle='--')
                ax.axhline(-ROLL_PITCH_THRESHOLD, color='orange', linestyle='--')
            ax.legend()

    def update(frame):
        for i, data in enumerate(data_sources):
            if isinstance(data, tuple):
                lines[2 * i].set_data(range(len(data[0])), data[0])
                lines[2 * i + 1].set_data(range(len(data[1])), data[1])
                lines[2 * i].axes.relim()
                lines[2 * i].axes.autoscale_view()
            else:
                lines[i].set_data(range(len(data)), data)
                lines[i].axes.relim()
                lines[i].axes.autoscale_view()
        return lines

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    t = threading.Thread(target=run_diagnostics)
    t.daemon = True
    t.start()
    plot_live()
