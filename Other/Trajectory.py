import time
import threading
import logging
import numpy as np
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig

current_x = 0.0
current_y = 0.0
current_z = 0.0

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
    global current_x, current_y, current_z
    current_x = data.get('stateEstimate.x', current_x)
    current_y = data.get('stateEstimate.y', current_y)
    current_z = data.get('stateEstimate.z', current_z)

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def bezier_curve(P0, P1, P2, steps):
    t_values = np.linspace(0, 1, steps)
    curve = []
    for t in t_values:
        x = (1 - t)**2 * P0[0] + 2 * (1 - t) * t * P1[0] + t**2 * P2[0]
        y = (1 - t)**2 * P0[1] + 2 * (1 - t) * t * P1[1] + t**2 * P2[1]
        curve.append((x, y))
    return curve

def plot_curve(P0, P1, P2, curve):
    xs, ys = zip(*curve)
    plt.plot(xs, ys, label="Bézier Curve")
    plt.plot([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], 'ro--', label="Control Points")
    plt.title("Planned Bézier Trajectory")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Main
def main():
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()
    URI = 'radio://0/80/2M'

    target_altitude = 1.0
    takeoff_duration = 2.0
    hover_duration = 3.0
    go_to_duration = 0.2

    pid_x = PIDController(kp=0.05, ki=0.002, kd=0.02)
    pid_y = PIDController(kp=0.05, ki=0.002, kd=0.02)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        log_conf = LogConfig(name='StateEstimate', period_in_ms=50)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_thread = threading.Thread(target=log_thread_func, args=(scf, log_conf))
        log_thread.daemon = True
        log_thread.start()

        commander = HighLevelCommander(scf.cf)

        # Takeoff and initial hover
        commander.takeoff(target_altitude, takeoff_duration)
        time.sleep(takeoff_duration + 1.0)

        setpoint_x = current_x
        setpoint_y = current_y
        pid_x.setpoint = setpoint_x
        pid_y.setpoint = setpoint_y

        print(f"Hovering at initial position: x = {setpoint_x:.2f}, y = {setpoint_y:.2f}")
        hover_start = time.time()
        last_time = hover_start
        while time.time() - hover_start < hover_duration:
            now = time.time()
            dt = now - last_time
            last_time = now

            corr_x = pid_x.update(current_x, dt)
            corr_y = pid_y.update(current_y, dt)
            new_x = setpoint_x + corr_x
            new_y = setpoint_y + corr_y

            commander.go_to(new_x, new_y, target_altitude, yaw=0.0, duration_s=go_to_duration, relative=False)
            time.sleep(0.05)

        # Curved flight path
        P0 = (setpoint_x, setpoint_y)
        P2 = (setpoint_x + 1.5, setpoint_y + 1.0)  # Destination
        P1 = (setpoint_x + 0.75, setpoint_y + 1.5)  # Control point

        print("Generating curved trajectory...")
        curve_points = bezier_curve(P0, P1, P2, steps=40)
        plot_curve(P0, P1, P2, curve_points)

        print("Flying along Bézier curve...")
        for x, y in curve_points:
            commander.go_to(x, y, target_altitude, yaw=0.0, duration_s=go_to_duration, relative=False)
            time.sleep(go_to_duration)

        # Hover at destination
        print("Hovering at destination...")
        pid_x.setpoint = P2[0]
        pid_y.setpoint = P2[1]
        hover_start = time.time()
        last_time = hover_start
        while time.time() - hover_start < hover_duration:
            now = time.time()
            dt = now - last_time
            last_time = now
            corr_x = pid_x.update(current_x, dt)
            corr_y = pid_y.update(current_y, dt)
            new_x = P2[0] + corr_x
            new_y = P2[1] + corr_y
            commander.go_to(new_x, new_y, target_altitude, yaw=0.0, duration_s=go_to_duration, relative=False)
            time.sleep(0.05)

        # Land
        print("Landing...")
        commander.land(0.0, 2.0)
        time.sleep(3.0)
        print("Landed.")

if __name__ == '__main__':
    main()
