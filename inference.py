import torch
import cv2
from act_policy import ACTPolicy

state_dim = 14
action_dim = 7

model = ACTPolicy(state_dim,action_dim)

model.load_state_dict(
    torch.load("models/act_policy.pth")
)

model.eval()

def predict(image,state):

    img = cv2.resize(image,(224,224))
    img = img.transpose(2,0,1)/255.0

    img = torch.tensor(img).unsqueeze(0).float()
    state = torch.tensor(state).unsqueeze(0).float()

    actions = model(img,state)

    return actions.detach().numpy()