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

# Global state variables
current_x = 0.0
current_y = 0.0

# Data for live plotting
trajectory_x = []
trajectory_y = []
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
    global current_x, current_y
    current_x = data.get('stateEstimate.x', current_x)
    current_y = data.get('stateEstimate.y', current_y)

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            if not plot_running:
                break
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def plot_live():
    fig, ax = plt.subplots()
    ax.set_title("Live XY Trajectory")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)

    line, = ax.plot([], [], 'b-', label="Trajectory")
    dot, = ax.plot([], [], 'ro', label="Current Position")
    ax.legend()

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def update(frame):
        if trajectory_x and trajectory_y:
            line.set_data(trajectory_x, trajectory_y)
            dot.set_data(trajectory_x[-1], trajectory_y[-1])
        return line, dot

    ani = animation.FuncAnimation(fig, update, init_func=init, interval=100, blit=True)
    plt.show()


def follow_path(commander, pid_x, pid_y, altitude, duration, dt=0.05):
    global trajectory_x, trajectory_y

    t_start = time.time()
    last_time = t_start

    # Parameters for figure-eight path
    radius = 0.3
    speed = 0.6  # radians per second

    while time.time() - t_start < duration:
        now = time.time()
        elapsed = now - last_time
        t = now - t_start
        last_time = now

        # Figure-eight parametric equations
        path_x = radius * np.sin(speed * t)
        path_y = radius * np.sin(speed * t) * np.cos(speed * t)

        pid_x.setpoint = path_x
        pid_y.setpoint = path_y

        dx = pid_x.update(current_x, elapsed)
        dy = pid_y.update(current_y, elapsed)

        commander.go_to(current_x + dx, current_y + dy, altitude, yaw=0.0, duration_s=0.2, relative=False)

        trajectory_x.append(current_x)
        trajectory_y.append(current_y)

        time.sleep(dt)

def reset_kalman(scf):
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(0.5)

def main():
    global plot_running
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()

    URI = 'radio://0/80/2M'
    target_altitude = 1.0
    flight_duration = 15.0

    pid_x = PIDController(kp=0.2, ki=0.0, kd=0.01)
    pid_y = PIDController(kp=0.2, ki=0.0, kd=0.01)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        log_conf = LogConfig(name='StateEstimate', period_in_ms=50)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')

        log_thread = threading.Thread(target=log_thread_func, args=(scf, log_conf))
        log_thread.daemon = True
        log_thread.start()

        plot_thread = threading.Thread(target=plot_live)
        plot_thread.daemon = True
        plot_thread.start()

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

if __name__ == '__main__':
    main()