import time
import logging

# This code is to check the position when the drone remains static
import cflib.crtp  # low-level drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig

# Set logging level if desired
logging.basicConfig(level=logging.INFO)

# Replace this URI with the one matching your Crazyflie radio settings.
URI = 'radio://0/80/2M'

def log_callback(timestamp, data, logconf):
    """
    Callback for processing log data.
    This function is called every time new data is received from the Crazyflie.
    """
    # Format the received data and print it
    x = data.get('stateEstimate.x', None)
    y = data.get('stateEstimate.y', None)
    z = data.get('stateEstimate.z', None)
    if x is not None and y is not None and z is not None:
        print(f"[{timestamp} ms] x: {x:.2f}, y: {y:.2f}, z: {z:.2f}")

def main():
    # Initialize the low-level drivers (ensure radio communication is ready).
    cflib.crtp.init_drivers()

    # Create a logging configuration that requests the Crazyflie's estimated state.
    log_conf = LogConfig(name='StateEstimate', period_in_ms=100)
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')

    # Use SyncCrazyflie to manage the connection lifecycle.
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        # Subscribe to the logging data stream using SyncLogger.
        with SyncLogger(scf, log_conf) as logger:
            print("Starting to log real-time position. Press Ctrl+C to stop.")
            try:
                # Process log entries as they come in.
                for log_entry in logger:
                    timestamp, data, logconf = log_entry
                    log_callback(timestamp, data, logconf)
                    # Optional delay if needed
                    time.sleep(0.1)
            except KeyboardInterrupt:
                # Clean shutdown on user interruption.
                print("Stopping log...")

if __name__ == '__main__':
    main()

