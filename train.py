import torch
from torch.utils.data import DataLoader
from dataset_loader import ACTDataset
from act_policy import ACTPolicy
import torch.nn as nn
from tqdm import tqdm

dataset = ACTDataset("dataset")

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

state_dim = 14
action_dim = 7

model = ACTPolicy(state_dim,action_dim)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.MSELoss()

EPOCHS = 50

for epoch in range(EPOCHS):

    total_loss = 0

    for img,state,action in tqdm(loader):

        pred = model(img,state)

        loss = loss_fn(pred,action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",total_loss)

torch.save(model.state_dict(),"models/act_policy.pth")