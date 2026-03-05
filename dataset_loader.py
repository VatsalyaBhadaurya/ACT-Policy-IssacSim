import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset

CHUNK = 20

class ACTDataset(Dataset):

    def __init__(self, root="dataset"):

        self.samples = []

        for ep in os.listdir(root):

            ep_dir = os.path.join(root, ep)

            states = np.load(f"{ep_dir}/states.npy")
            actions = np.load(f"{ep_dir}/actions.npy")

            img_dir = f"{ep_dir}/images"

            for i in range(len(states) - CHUNK):

                img_path = f"{img_dir}/{i:04d}.png"

                self.samples.append(
                    (img_path, states, actions, i)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, states, actions, i = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))
        img = img.transpose(2,0,1) / 255.0

        state = states[i]
        action_chunk = actions[i:i+CHUNK]

        return (
            torch.tensor(img).float(),
            torch.tensor(state).float(),
            torch.tensor(action_chunk).float()
        )