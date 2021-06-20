import gym
import torch
import torchvision
import random

games = {
    'breakout' : 'BreakoutDeterministic-v4',
    'space invaders' : 'SpaceInvadersDeterministic-v4',
    'pong' : 'PongDeterministic-v4',
    'centipede' : 'CentipedeDeterministic-v4'
}

class Environment:
    def __init__(self, game='breakout'):
        if game.lower() not in games:
            raise ValueError(f"Game {game} not supported by this environment.")

        self.env = gym.make(games[game.lower()])

        self.ale = self.env.ale
        self.spec = self.env.spec
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range

        self.history = torch.zeros((4, 84, 84))
        self.noop_steps = 10

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(),
            lambda x : torchvision.transforms.functional.crop(x, 25, 8, 180, 144),
            torchvision.transforms.Resize((84, 84), 0),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean = mean, std = std)
        ])

    def reset(self, eval=False):
        self.env.reset()
        self.state, _, _, _ = self.step(1) # experiment to start the epoch faster. Could also tile zeros.

        if eval:
            for _ in range(random.randint(1, self.noop_steps)):
                _, _, _, _ = self.env.step(1)

    def render(self, mode=None):
        return self.env.render(mode)

    def clip(self, reward):
        return min(reward, 1)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        clipped_reward = self.clip(reward)
        info['unclipped_reward'] = reward

        self.history = torch.cat([self.history[1:], self.transforms(observation)])

        return self.history.unsqueeze(0), clipped_reward, done, info

    def close(self):
        return self.env.close()
