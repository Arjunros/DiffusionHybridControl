import torch
import numpy as np
from tqdm import tqdm
from diffusion_policy import DiffusionPolicy

data = np.load('expert_data.npy', allow_pickle=True)
states = np.array([d[0] for d in data])
actions = np.array([d[1] for d in data])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DiffusionPolicy(states.shape[1], actions.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in tqdm(range(500)):
    idx = np.random.randint(0, len(states), 128)
    s = torch.tensor(states[idx], dtype=torch.float32).to(device)
    a = torch.tensor(actions[idx], dtype=torch.float32).to(device)
    t = torch.rand((s.shape[0], 1), device=device)

    noise = torch.randn_like(a)
    noisy_a = a + noise * t
    pred_noise = model(torch.cat([s, noisy_a], dim=-1), t)
    loss = ((pred_noise - noise) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'diffusion_policy.pth')
print("Diffusion policy trained and saved.")
