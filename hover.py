import time

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

def activate_mellinger_controller(scf, use_mellinger):
    controller = 1
    if use_mellinger:
        controller = 2
    scf.cf.param.set_value('stabilizer.controller', controller)

def arm(scf):
    scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)

def run_hover_sequence(scf):
    activate_mellinger_controller(scf, False)

    commander = scf.cf.high_level_commander
    commander.takeoff(1.0, 2.0)
    time.sleep(3)

    print("Hovering...")
    time.sleep(12)

    print("Landing...")
    commander.land(0.0, 2.0)
    time.sleep(3)

    commander.stop()

uris = {
    'radio://0/80/2M'
}

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.reset_estimators()
        swarm.parallel_safe(arm)
        swarm.parallel_safe(run_hover_sequence)
