import mujoco
import mujoco_viewer
import numpy as np
import time


class RoboticArmEnv:
    def __init__(self, xml_path="2R_robotic_arm.xml"):
        """Initialize the MuJoCo environment."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.viewer = None  # viewer handle

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        return self.get_state()

    def get_state(self):
        """Return current state vector (joint positions + velocities)."""
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, control_signal):
        """Apply control signal and step simulation."""
        self.data.ctrl[:] = control_signal
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()
        time.sleep(0.01)

    def close(self):
        """Close viewer gracefully."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
