from env_robotic_arm import RoboticArmEnv
from predictive_sampling import PredictiveSampler
from diffusion_policy import DiffusionPolicy
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Initialize environment and controllers ---
env = RoboticArmEnv()
sampler = PredictiveSampler(env)
model = DiffusionPolicy(state_dim=4, action_dim=2)
model.load_state_dict(torch.load('diffusion_policy.pth'))
model.eval()

# --- Initialize state and goal ---
state = env.reset()
goal = np.array([0.5, 0.3])

# --- Data logging arrays ---
qpos_log = []
qvel_log = []
ctrl_log = []
ee_pos_log = []

# --- Hybrid Control Loop ---
steps = 1000
for t in range(steps):
    # Predictive sampling generates initial action sequence
    u_pred, seq = sampler.sample_trajectory(state, goal)

    # Convert state/action to tensors for diffusion model
    s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    a_tensor = torch.tensor(u_pred, dtype=torch.float32).unsqueeze(0)
    t_tensor = torch.rand((1, 1))
    
    # Refine predictive action with diffusion policy
    refined_action = (
        a_tensor + model(torch.cat([s_tensor, a_tensor], dim=-1), t_tensor)
    ).detach().numpy()[0]

    # Apply refined action
    state = env.step(refined_action)
    env.render()

    # --- Record Data ---
    qpos_log.append(env.data.qpos.copy())
    qvel_log.append(env.data.qvel.copy())
    ctrl_log.append(refined_action.copy())

    # Compute end-effector position (forward kinematics)
    x = 0.3 * np.cos(env.data.qpos[0]) + 0.3 * np.cos(env.data.qpos[0] + env.data.qpos[1])
    y = 0.3 * np.sin(env.data.qpos[0]) + 0.3 * np.sin(env.data.qpos[0] + env.data.qpos[1])
    ee_pos_log.append([x, y])

# --- Keep the viewer open after simulation ---
print("Hybrid control simulation finished.")
print("Viewer will remain open. Press Ctrl+C in terminal to stop.")

try:
    while True:
        env.render()
except KeyboardInterrupt:
    env.close()
    print("\n🟢 Viewer closed by user.")

# --- Convert logs to NumPy arrays ---
qpos_log = np.array(qpos_log)
qvel_log = np.array(qvel_log)
ctrl_log = np.array(ctrl_log)
ee_pos_log = np.array(ee_pos_log)
time_vec = np.linspace(0, steps * env.dt, steps)

# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time_vec, qpos_log[:, 0], label='Joint 1 Angle')
plt.plot(time_vec, qpos_log[:, 1], label='Joint 2 Angle')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_vec, qvel_log[:, 0], label='Joint 1 Velocity')
plt.plot(time_vec, qvel_log[:, 1], label='Joint 2 Velocity')
plt.ylabel('Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_vec, ctrl_log[:, 0], label='Motor 1 Torque')
plt.plot(time_vec, ctrl_log[:, 1], label='Motor 2 Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- End-effector trajectory plot ---
plt.figure(figsize=(6, 6))
plt.plot(ee_pos_log[:, 0], ee_pos_log[:, 1], 'b-', label='End-effector Path')
plt.scatter(ee_pos_log[0, 0], ee_pos_log[0, 1], color='green', label='Start')
plt.scatter(ee_pos_log[-1, 0], ee_pos_log[-1, 1], color='red', label='End')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('End-Effector Trajectory')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
