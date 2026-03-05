import torch
import torch.nn as nn
import torchvision.models as models

CHUNK = 20

class ACTPolicy(nn.Module):

    def __init__(self,state_dim,action_dim):

        super().__init__()

        resnet = models.resnet18(weights=None)

        self.vision = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.state_embed = nn.Linear(state_dim,256)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8
            ),
            num_layers=4
        )

        self.fc = nn.Linear(256,action_dim*CHUNK)

    def forward(self,image,state):

        v = self.vision(image).flatten(1)

        s = self.state_embed(state)

        x = v + s

        x = self.transformer(x.unsqueeze(0)).squeeze(0)

        out = self.fc(x)

        return out.view(-1,CHUNK)