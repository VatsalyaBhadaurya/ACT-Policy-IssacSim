import torch
import cv2
import numpy as np

from act_policy import ACTPolicy
from record_demo import record_step, save_episode

# --------------------------------
# Robot + camera placeholders
# --------------------------------

class IsaacRobotInterface:
    
    def __init__(self):
        self.state_dim = 14
        self.action_dim = 7

    def get_camera_frame(self):
        """
        Replace with Isaac camera sensor call
        """
        frame = np.zeros((480,640,3),dtype=np.uint8)
        return frame

    def get_robot_state(self):
        """
        Replace with joint positions from robot
        """
        state = np.random.rand(self.state_dim)
        return state

    def apply_action(self,action):
        """
        Send action to robot controller
        """
        print("Applying action:",action)


# --------------------------------
# Load ACT policy
# --------------------------------

def load_policy():

    state_dim = 14
    action_dim = 7

    model = ACTPolicy(state_dim,action_dim)

    model.load_state_dict(
        torch.load("models/act_policy.pth")
    )

    model.eval()

    return model


# --------------------------------
# preprocess image
# --------------------------------

def preprocess(img):

    img = cv2.resize(img,(224,224))
    img = img.transpose(2,0,1)
    img = img / 255.0

    return torch.tensor(img).unsqueeze(0).float()


# --------------------------------
# inference loop
# --------------------------------

def run_policy():

    robot = IsaacRobotInterface()
    model = load_policy()

    while True:

        frame = robot.get_camera_frame()
        state = robot.get_robot_state()

        img_tensor = preprocess(frame)
        state_tensor = torch.tensor(state).unsqueeze(0).float()

        actions = model(img_tensor,state_tensor)

        actions = actions.detach().numpy()[0]

        for action in actions:

            robot.apply_action(action)


# --------------------------------
# demo recording loop
# --------------------------------

def record_demos():

    robot = IsaacRobotInterface()

    episode = 0

    while episode < 10:

        for step in range(100):

            frame = robot.get_camera_frame()
            state = robot.get_robot_state()

            # example teleop action
            action = np.random.rand(7)

            record_step(frame,state,action)

        save_episode(episode)

        episode += 1


if __name__ == "__main__":

    mode = input("mode (record/run): ")

    if mode == "record":
        record_demos()

    if mode == "run":
        run_policy()