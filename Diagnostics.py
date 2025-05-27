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

# Data lists for plotting
alt_data, vz_data, thrust_data = [], [], []
roll_data, pitch_data, yaw_data = [], [], []

# Max data points to display
WINDOW = 200
plot_running = True

URI = 'radio://0/80/2M'

THRUST_MAX = 50000
ROLL_PITCH_THRESHOLD = 30.0  # degrees


def log_callback(timestamp, data, logconf):
    global altitude, velocity_z, thrust, roll, pitch, yaw
    altitude = data.get('stateEstimate.z', altitude)
    velocity_z = data.get('stateEstimate.vz', velocity_z)
    thrust = data.get('stabilizer.thrust', thrust)
    roll = data.get('stabilizer.roll', roll)
    pitch = data.get('stabilizer.pitch', pitch)
    yaw = data.get('stabilizer.yaw', yaw)

    alt_data.append(altitude)
    vz_data.append(velocity_z)
    thrust_data.append(thrust)
    roll_data.append(roll)
    pitch_data.append(pitch)
    yaw_data.append(yaw)

    # Trim data lists
    if len(alt_data) > WINDOW:
        alt_data.pop(0)
        vz_data.pop(0)
        thrust_data.pop(0)
        roll_data.pop(0)
        pitch_data.pop(0)
        yaw_data.pop(0)


def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            if not plot_running:
                break
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)


def plot_live():
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle("Drone Diagnostics")

    lines = []
    titles = ["Altitude (z)", "Vertical Velocity (vz)", "Thrust",
              "Roll", "Pitch", "Yaw"]
    data_sources = [alt_data, vz_data, thrust_data, roll_data, pitch_data, yaw_data]

    for i, (ax, title) in enumerate(zip(axs.flat, titles)):
        ax.set_title(title)
        ax.set_xlim(0, WINDOW)
        ax.grid(True)
        line, = ax.plot([], [], label=title)

        # Add threshold overlays
        if title == "Thrust":
            ax.axhline(THRUST_MAX, color='r', linestyle='--', label='Thrust Max')
        elif title in ["Roll", "Pitch"]:
            ax.axhline(ROLL_PITCH_THRESHOLD, color='orange', linestyle='--', label='±30°')
            ax.axhline(-ROLL_PITCH_THRESHOLD, color='orange', linestyle='--')

        ax.legend()
        lines.append(line)

    def update(frame):
        for line, data in zip(lines, data_sources):
            line.set_data(range(len(data)), data)
            line.axes.relim()
            line.axes.autoscale_view()
        return lines

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def reset_kalman(scf):
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(0.5)


def flight_test_sequence(commander):
    print("[TEST] Takeoff")
    commander.takeoff(0.5, 2.0)
    time.sleep(3.0)

    print("[TEST] Hovering")
    time.sleep(2.0)

    print("[TEST] Increase altitude")
    commander.go_to(0.0, 0.0, 1.3, 0.0, 2.0, False)
    time.sleep(2.5)

    print("[TEST] Decrease altitude")
    commander.go_to(0.0, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)

    print("[TEST] Forward")
    commander.go_to(0.5, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)

    print("[TEST] Backward")
    commander.go_to(0.0, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)

    print("[TEST] Left")
    commander.go_to(0.0, 0.5, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)

    print("[TEST] Right")
    commander.go_to(0.0, 0.0, 1.0, 0.0, 2.0, False)
    time.sleep(2.5)

    print("[TEST] Yaw rotation")
    commander.go_to(0.0, 0.0, 1.0, 180.0, 3.0, False)
    time.sleep(3.0)

    print("[TEST] Return yaw")
    commander.go_to(0.0, 0.0, 1.0, 0.0, 3.0, False)
    time.sleep(3.0)

    print("[TEST] Landing")
    commander.land(0.0, 2.0)
    time.sleep(3.0)


def run_diagnostics():
    global plot_running
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        log_conf = LogConfig(name='FullDiagnostics', period_in_ms=50)
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stateEstimate.vz', 'float')
        log_conf.add_variable('stabilizer.thrust', 'uint16_t')
        log_conf.add_variable('stabilizer.roll', 'float')
        log_conf.add_variable('stabilizer.pitch', 'float')
        log_conf.add_variable('stabilizer.yaw', 'float')

        threading.Thread(target=log_thread_func, args=(scf, log_conf), daemon=True).start()

        commander = HighLevelCommander(scf.cf)

        reset_kalman(scf)
        flight_test_sequence(commander)
        plot_running = False

if __name__ == '__main__':
    t = threading.Thread(target=run_diagnostics)
    t.daemon = True
    t.start()
    plot_live()
