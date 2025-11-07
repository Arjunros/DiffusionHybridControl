import numpy as np
from env_robotic_arm import RoboticArmEnv
from predictive_sampling import PredictiveSampler

env = RoboticArmEnv()
sampler = PredictiveSampler(env)
data = []

for episode in range(100):
    state = env.reset()
    goal = np.array([0.5, 0.3])
    for t in range(50):
        u, seq = sampler.sample_trajectory(state, goal)
        next_state = env.step(u)
        data.append((state, u))
        state = next_state

# Save as object array
np.save('expert_data.npy', np.array(data, dtype=object))
print("Expert data saved successfully as 'expert_data.npy'")
