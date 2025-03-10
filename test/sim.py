#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import time
from mppii import ROOT

def main():
    # Load the MuJoCo model from the XML file.
    model = mujoco.MjModel.from_xml_path(ROOT + "/models/franka_emika_panda/mjx_scene.xml")
    data = mujoco.MjData(model)

    # Simulate indefinitely.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Track simulation time
        sim_start = time.time()
        while viewer.is_running():
            step_start = time.time()
            
            mujoco.mj_step(model, data)  # Advance the simulation by one step.
            viewer.sync()
            
            # Wait until real time catches up with simulation time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
