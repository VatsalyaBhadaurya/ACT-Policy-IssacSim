import os
import cv2
import numpy as np

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

images = []
states = []
actions = []

def record_step(image, state, action):
    images.append(image)
    states.append(state)
    actions.append(action)

def save_episode(ep):

    ep_dir = f"{DATASET_DIR}/episode_{ep}"
    os.makedirs(ep_dir, exist_ok=True)

    np.save(f"{ep_dir}/states.npy", np.array(states))
    np.save(f"{ep_dir}/actions.npy", np.array(actions))

    img_dir = f"{ep_dir}/images"
    os.makedirs(img_dir, exist_ok=True)

    for i, img in enumerate(images):
        cv2.imwrite(f"{img_dir}/{i:04d}.png", img)

    print("Saved episode", ep)