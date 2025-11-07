import numpy as np

class PredictiveSampler:
    def __init__(self, env, horizon=20, num_samples=100):
        self.env = env
        self.horizon = horizon
        self.num_samples = num_samples

    def cost_function(self, trajectory, goal):
        pos_error = np.linalg.norm(trajectory[-1][:2] - goal)
        smoothness = np.sum(np.diff(trajectory[:, -2:])**2)
        return pos_error + 0.001 * smoothness


    def simulate(self, init_state, actions):
        trajectory = [init_state]
        state = init_state.copy()
        for a in actions:
            next_state = state + 0.01 * np.concatenate([a, np.zeros_like(a)])
            trajectory.append(next_state)
            state = next_state
        return np.array(trajectory)

    def sample_trajectory(self, current_state, goal):
        best_cost = float('inf')
        best_seq = None

        for _ in range(self.num_samples):
            actions = np.random.uniform(-5, 5, (self.horizon, 2))

            traj = self.simulate(current_state, actions)
            cost = self.cost_function(traj, goal)
            if cost < best_cost:
                best_cost = cost
                best_seq = actions

        return best_seq[0], best_seq  # first action and full sequence
