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
current_x = 0.0
current_y = 0.0
current_z = 0.0
trajectory_x = []
trajectory_y = []
trajectory_z = []
range_distances = {"Front": 0, "Back": 0, "Left": 0, "Right": 0, "Up": 0, "Down": 0}
plot_running = True

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limit=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_error = 0.0
        self.output_limit = output_limit

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return max(min(output, self.output_limit), -self.output_limit)

def log_callback(timestamp, data, logconf):
    global current_x, current_y, current_z
    current_x = data.get('stateEstimate.x', current_x)
    current_y = data.get('stateEstimate.y', current_y)
    current_z = data.get('stateEstimate.z', current_z)

    trajectory_x.append(current_x)
    trajectory_y.append(current_y)
    trajectory_z.append(current_z)

    range_distances["Front"] = data.get('range.front', 0) / 1000.0
    range_distances["Back"] = data.get('range.back', 0) / 1000.0
    range_distances["Left"] = data.get('range.left', 0) / 1000.0
    range_distances["Right"] = data.get('range.right', 0) / 1000.0
    range_distances["Up"] = data.get('range.up', 0) / 1000.0
    range_distances["Down"] = data.get('range.zrange', 0) / 1000.0

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            if not plot_running:
                break
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def follow_path(commander, pid_x, pid_y, altitude, duration, dt=0.05):
    t_start = time.time()
    last_time = t_start

    radius = 0.3
    speed = 0.6

    while time.time() - t_start < duration:
        now = time.time()
        elapsed = now - last_time
        t = now - t_start
        last_time = now

        path_x = radius * np.sin(speed * t)
        path_y = radius * np.sin(speed * t) * np.cos(speed * t)

        pid_x.setpoint = path_x
        pid_y.setpoint = path_y

        dx = pid_x.update(current_x, elapsed)
        dy = pid_y.update(current_y, elapsed)

        commander.go_to(current_x + dx, current_y + dy, altitude, yaw=0.0, duration_s=0.2, relative=False)

        time.sleep(dt)

def reset_kalman(scf):
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(0.5)

def drone_flight():
    global plot_running
    URI = 'radio://0/80/2M'
    target_altitude = 1.0
    flight_duration = 15.0

    pid_x = PIDController(kp=0.2, ki=0.0, kd=0.01)
    pid_y = PIDController(kp=0.2, ki=0.0, kd=0.01)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        log_conf = LogConfig(name='State', period_in_ms=50)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('range.front', 'uint16_t')
        log_conf.add_variable('range.back', 'uint16_t')
        log_conf.add_variable('range.left', 'uint16_t')
        log_conf.add_variable('range.right', 'uint16_t')
        log_conf.add_variable('range.up', 'uint16_t')
        log_conf.add_variable('range.zrange', 'uint16_t')

        threading.Thread(target=log_thread_func, args=(scf, log_conf), daemon=True).start()

        commander = HighLevelCommander(scf.cf)

        try:
            reset_kalman(scf)
            print("Taking off...")
            commander.takeoff(target_altitude, 1.0)
            time.sleep(2.0)

            print("Following path...")
            follow_path(commander, pid_x, pid_y, target_altitude, duration=flight_duration)

        finally:
            print("Landing...")
            commander.land(0.0, 2.0)
            time.sleep(3.0)
            scf.cf.commander.send_stop_setpoint()
            plot_running = False
            print("Done.")

def plot_live():
    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.set_title("Live 3D Trajectory")
    ax3d.set_xlim([-1, 1])
    ax3d.set_ylim([-1, 1])
    ax3d.set_zlim([0, 2])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    traj_line, = ax3d.plot([], [], [], lw=2)

    ax2d = fig.add_subplot(122)
    ax2d.set_title("Multiranger Readings")
    bar_labels = ["Front", "Back", "Left", "Right", "Up", "Down"]
    bar_values = [0] * 6
    bars = ax2d.bar(bar_labels, bar_values)
    ax2d.set_ylim(0, 2)

    def update(frame):
        if trajectory_x and trajectory_y and trajectory_z:
            traj_line.set_data(trajectory_x, trajectory_y)
            traj_line.set_3d_properties(trajectory_z)
        for i, label in enumerate(bar_labels):
            bars[i].set_height(range_distances[label])
        return [traj_line, *bars]

    ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

def main():
    cflib.crtp.init_drivers()  # ✅ Make sure drivers are initialized in main thread

    drone_thread = threading.Thread(target=drone_flight)
    drone_thread.daemon = True
    drone_thread.start()

    plot_live()  # Runs in main thread

if __name__ == '__main__':
    main()
