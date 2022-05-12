import os, random
import numpy as np
import torch
from torch import nn
import itertools
from baselines_wrappers import DummyVecEnv
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time
import argparse

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten())

    # Compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out

class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device

        conv_net = nature_cnn(env.observation_space)

        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions
    
    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()}

        self.load_state_dict(params)

class DuelingNetwork(Network):
    def __init__(self, env, device, network_type="dueling"):
        super().__init__(env, device, network_type)

        self.num_actions = env.action_space.n

        self.device = device

        self.network_type = network_type

        self.depths = (32, 64, 64)

        self.final_layer = 512

        n_input_channels = env.observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, self.depths[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.depths[0], self.depths[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.depths[1], self.depths[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(env.observation_space.sample()[None]).float()).shape[1]

        self.fc_value = nn.Sequential(
            nn.Linear(n_flatten, self.final_layer),
            nn.ReLU()
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(n_flatten, self.final_layer),
            nn.ReLU()
        )

        self.value_final = nn.Linear(self.final_layer, 1)
        self.adv_final = nn.Linear(self.final_layer, self.num_actions)

    def forward(self, x):
        x = self.cnn(x)

        value = self.fc_value(x)
        adv = self.fc_adv(x)

        value = self.value_final(value)
        adv = self.adv_final(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)

        Q = value + adv - advAverage

        return Q

parser = argparse.ArgumentParser(description="parameters for training")
parser.add_argument('--network_type', default="vanilla", type=str)
parser.add_argument('--save_path', default="./breakout_dqn/atari_model_pack", type=str)
parser.add_argument('--game_to_play', default="BreakoutNoFrameskip-v4", type=str)

args = parser.parse_args()

SAVE_PATH = args.save_path
NETWORK_TYPE = args.network_type
GAME_TO_PLAY = args.game_to_play

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

make_env = lambda: make_atari_deepmind(GAME_TO_PLAY, observe=True)

vec_env = DummyVecEnv([make_env for _ in range(1)])

env = BatchedPytorchFrameStack(vec_env, k=4)

if NETWORK_TYPE == "dueling":
    net = DuelingNetwork(env, device)
else:
    net = Network(env, device)
    
net = net.to(device)

net.load(SAVE_PATH)

obs = env.reset() 
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, _ = env.step(action)
    env.render(mode='rgb_array')
    time.sleep(0.02)

    if done[0]:
        obs = env.reset()
        beginning_episode = True