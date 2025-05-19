import time
import threading
import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
import matplotlib.pyplot as plt
import numpy as np

# Global state variables
current_x = 0.0
current_y = 0.0
current_z = 0.0

# Record the trajectory
trajectory_x = []
trajectory_y = []
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
    global current_x, current_y, current_z
    current_x = data.get('stateEstimate.x', current_x)
    current_y = data.get('stateEstimate.y', current_y)
    current_z = data.get('stateEstimate.z', current_z)

def log_thread_func(scf, log_conf):
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def hover(commander, pid_x, pid_y, setpoint_x, setpoint_y, altitude, duration, dt=0.05):
    print(f"Hovering at x={setpoint_x:.2f}, y={setpoint_y:.2f} for {duration}s")
    start = time.time()
    last_time = start
    while time.time() - start < duration:
        now = time.time()
        elapsed = now - last_time
        last_time = now

        corr_x = pid_x.update(current_x, elapsed)
        corr_y = pid_y.update(current_y, elapsed)
        new_x = setpoint_x + corr_x
        new_y = setpoint_y + corr_y

        commander.go_to(new_x, new_y, altitude, yaw=0.0, duration_s=0.2, relative=False)

        print(f"[{time.time()-start:4.1f}s] Pos: x={current_x:.2f}, y={current_y:.2f} | "
              f"Target: x={new_x:.2f}, y={new_y:.2f} | PID: dx={corr_x:.3f}, dy={corr_y:.3f}")

        time.sleep(dt)

# Recording process
def record_thread_func():
    global recording
    while recording:
        trajectory_x.append(current_x)
        trajectory_y.append(current_y)
        time.sleep(0.05)

def main():
    global recording
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()
    URI = 'radio://0/80/2M'

    target_altitude = 0.5
    takeoff_duration = 1.0
    initial_hover_duration = 10.0
    second_hover_duration = 6.0
    third_hover_duration = 6.0
    travel_duration = 1  # time to travel to new location

    # Set a fixed function 
    f_x = np.linspace(0, 1, 5)  
    f_y = f_x**2
    fun_sm = np.column_stack((f_x, f_y))
    # Set target offset for second location (relative)
    # second_location_offset = (0.5, 0.5)  # 0.5m forward, 0.5m right
    # third_location_offset = (-0.5, -0.5)  # time to hover at new location

    
    pid_x = PIDController(kp=0.05, ki=0.002, kd=0.02)
    pid_y = PIDController(kp=0.05, ki=0.002, kd=0.02)

    #with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
     #   log_conf = LogConfig(name='StateEstimate', period_in_ms=50)
     #   log_conf.add_variable('stateEstimate.x', 'float')
     #   log_conf.add_variable('stateEstimate.y', 'float')
     #  log_conf.add_variable('stateEstimate.z', 'float')
     #   log_thread = threading.Thread(target=log_thread_func, args=(scf, log_conf))
     #   log_thread.daemon = True
     #   log_thread.start()
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.set_value('kalman.resetEstimation', '1')
        scf.cf.param.set_value('kalman.initialX', str(current_x))
        scf.cf.param.set_value('kalman.initialY', str(current_y))
        scf.cf.param.set_value('kalman.initialZ', str(current_z))
        time.sleep(0.1)

        log_conf = LogConfig(name='StateEstimate', period_in_ms=50)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_thread = threading.Thread(target=log_thread_func, args=(scf, log_conf))
        log_thread.daemon = True
        log_thread.start()

        commander = HighLevelCommander(scf.cf)

        recording = True
        threading.Thread(target=record_thread_func, daemon=True).start()

        print("Takeoff...")
        commander.takeoff(target_altitude, takeoff_duration)
        time.sleep(takeoff_duration + 1.0)
        
        # Initial position
        initial_x = current_x
        initial_y = current_y
        pid_x.setpoint = initial_x
        pid_y.setpoint = initial_y

        # Hover at initial position
        hover(commander, pid_x, pid_y, initial_x, initial_y, target_altitude, initial_hover_duration)

        # # Move to second location
        # new_x = initial_x + second_location_offset[0]
        # new_y = initial_y + second_location_offset[1]
        # pid_x.setpoint = new_x
        # pid_y.setpoint = new_y

        # print(f"Flying to new location x={new_x:.2f}, y={new_y:.2f}...")
        # commander.go_to(new_x, new_y, target_altitude, yaw=0.0, duration_s=travel_duration, relative=False)
        # time.sleep(travel_duration + 1.0)

        # # Hover at new position
        # hover(commander, pid_x, pid_y, new_x, new_y, target_altitude, second_hover_duration)

        # # Move to third location
        # new_x = third_location_offset[0]
        # new_y = third_location_offset[1]
        # pid_x.setpoint = new_x
        # pid_y.setpoint = new_y

        # print(f"Flying to new location x={new_x:.2f}, y={new_y:.2f}...")
        # commander.go_to(new_x, new_y, target_altitude, yaw=0.0, duration_s=travel_duration, relative=False)
        # time.sleep(travel_duration + 1.0)

        # # Hover at new position
        # hover(commander, pid_x, pid_y, new_x, new_y, target_altitude, third_hover_duration)

        # Hover at the destination

        print("Landing...")
        commander.land(0.0, 2.0)
        time.sleep(3.0)
        print("Done.")
        recording = False

    plt.figure()
    plt.plot(trajectory_x, trajectory_y, label='Actual', alpha=0.7)
    plt.plot(f_x, f_y, '-o', label='Planned')
    plt.title('XY Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
