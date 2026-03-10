import genesis as gs
import torch
from your_act_policy import ACTPolicy  # Copy ACTPolicy class

gs.init(backend="cuda")

model = ACTPolicy()
model.load_state_dict(torch.load("act_policy_aloha.pt"))
model.eval()

scene = gs.Scene(show_viewer=True)
franka = scene.add_entity(gs.morphs.MJCF("franka.xml"))
scene.build()

obs = franka.get_observation()
while True:
    state = torch.tensor(obs["state"]).unsqueeze(0)
    action = model({"state": state})[0, 0].numpy()
    scene.step(action)
