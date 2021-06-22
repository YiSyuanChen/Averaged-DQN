import time
import gym
import torch
import numpy as np
import tensorflow as tf # BUG: should be import before torchvision
import torchvision
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from gym.wrappers import Monitor
import datetime
import os

from model import Model
from plotting import VisdomLinePlotter
from environment import Environment
from utils import Memory, EpsilonScheduler, make_log_dir, save_gif
import pdb

##### Arguments #####
parser = argparse.ArgumentParser(
    description='An implementation of the 2015 DeepMind DQN Paper')

parser.add_argument('--weights',
                    type=str,
                    help='weights file for pretrained weights')
parser.add_argument('--nosave',
                    default=False,
                    action='store_true',
                    help='do not save a record of the run')
parser.add_argument('--game',
                    type=str,
                    default='breakout')

args = parser.parse_args()

##### Hyper-Parameters #####
BATCH_SIZE = 32  # size of minibatch
MEM_SIZE = int(1e6)  # size of replay memory
TARGET_UPDATE_EVERY = 10000  # in units of minibatch updates
GAMMA = 0.99  # discount factor
UPDATE_FREQ = 4  # perform minibatch update once every UPDATE_FREQ
INIT_MEMORY_SIZE = 200000  # initial size of memory before minibatch updates begin
#scheduler = EpsilonScheduler(schedule=[(0, 1),(INIT_MEMORY_SIZE,1),(1e6, 0.1)])
scheduler = EpsilonScheduler(schedule=[(0, 1), (INIT_MEMORY_SIZE, 1), (2e6, 0.1), (30e6, 0.01)])


STORAGE_DEVICES = [
    'cuda:0'
]  # list of devices to use for episode storage (need about 10GB for 1 million memories)
DEVICE = 'cuda:0'  # list of devices for computation
EPISODES = int(1e5)  # total training episodes
NUM_TEST = 20
TEST_EVERY = 1000  # (episodes)
PLOT_EVERY = 10  # (episodes)
SAVE_EVERY = 1000  # (episodes)
EXPERIMENT_DIR = "experiments"
#GAME = 'breakout'
GAME = args.game

if not args.nosave:
    root_dir, weight_dir, video_dir = make_log_dir(EXPERIMENT_DIR, GAME)
    with open(os.path.join(EXPERIMENT_DIR, "current.txt"), "w") as f:
        f.write(os.path.abspath(video_dir))

##### Gym Environment #####
env = Environment(game=GAME)
mem = Memory(MEM_SIZE, storage_devices=STORAGE_DEVICES, target_device=DEVICE)

##### Q Functions #####
q_func = Model(env.action_space.n).to(DEVICE)
if args.weights:
    q_func.load_state_dict(torch.load(args.weights))

#target_q_func = Model(env.action_space.n).to(DEVICE)
#target_q_func.load_state_dict(q_func.state_dict())

# NEW: Build multiple target Q function
# Assume the last model in list is for step t-1
K = 10
target_q_func_list = []
for k in range(K):
    target_q_func = Model(env.action_space.n).to(DEVICE)
    target_q_func.load_state_dict(q_func.state_dict())
    target_q_func_list.append(target_q_func)

def avg_q_func(q_func_list, state):
    q_value_list = [q_func(state) for q_func in q_func_list]
    avg_q_value = sum(q_value_list) / len(q_value_list)
    return avg_q_value

##### Optimizer #####
#optimizer = optim.RMSprop(q_func.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
optimizer = optim.Adam(
    q_func.parameters(), lr=0.00001,
    eps=1.5e-4)  # 0.00001 for breakout, 0.00025 is faster for pong

##### Loss Function #####
loss_func = nn.SmoothL1Loss()

##### Logger #####
# for visdom
plot_title = "{} DQN ({})".format(
    GAME,
    datetime.datetime.now().strftime("%d/%m/%y %H:%M"))
plotter = VisdomLinePlotter(disable=args.nosave)

# for tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(os.path.join(root_dir,'logs'))

if not args.nosave:
    env = Monitor(env,
                  directory=video_dir,
                  video_callable=lambda count: count % 500 == 0,
                  force=True)

##### Testing #####

def test(active_target_count, save=False):
    print("[TESTING]")
    total_reward = 0
    unclipped_reward = 0

    for i in range(NUM_TEST):
        if i == 0 and save:
            frames = []

        env.reset(eval=True)  # performs random actions to start
        state, _, done, _ = env.step(env.action_space.sample())
        frame = 0

        while not done:
            if i == 0 and save:
                frames.append(state[0, 0])

            #q_values = q_func(state.to(DEVICE))
            # NEW: use average q value
            if active_target_count > 1:
                q_values = avg_q_func(target_q_func_list[-active_target_count+1:] + [q_func], state.to(DEVICE))
            else:
                q_values = avg_q_func([q_func], state.to(DEVICE))

            if np.random.random(
            ) > 0.01:  # small epsilon-greedy, sometimes 0.05
                action = torch.argmax(q_values, dim=1).item()
            else:
                action = env.action_space.sample()

            lives = env.ale.lives()
            next_state, reward, done, info = env.step(action)
            if env.ale.lives() != lives:  # lost life
                pass

            unclipped_reward += info['unclipped_reward']
            total_reward += reward
            state = next_state
            frame += 1

        if i == 0 and save:
            frames.append(state[0, 0])
            save_gif(
                frames, "{}.gif".format(
                    os.path.join(video_dir, str(scheduler.step_count()))))

    total_reward /= NUM_TEST
    unclipped_reward /= NUM_TEST

    with summary_writer.as_default():
        tf.summary.scalar('Total Test Reward', total_reward, step=scheduler.step_count())
        tf.summary.scalar('Total Unclipped Test Reward', unclipped_reward, step=scheduler.step_count())

    print(
        f"[TESTING] Total Reward: {total_reward}, Unclipped Reward: {unclipped_reward}"
    )

    return total_reward

##### Main Process #####
start_time = time.time()
active_target_count = 1 # NEW: for average
avg_reward = 0
avg_unclipped_reward = 0
avg_q = 0
num_parameter_updates = 0
for episode in range(EPISODES):
    avg_loss = 0
    total_reward = 0
    unclipped_reward = 0
    frame = 0

    env.reset()
    state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        q_values = q_func(state.to(DEVICE))
        if np.random.random() > scheduler.epsilon():  # epsilon-random policy
            action = torch.argmax(q_values, dim=1)
        else:
            action = env.action_space.sample()

        avg_q = 0.9 * avg_q + 0.1 * q_values.mean().item(
        )  # record average q value

        lives = env.ale.lives()  # get lives before action
        next_state, reward, done, info = env.step(action)
        reward = info['unclipped_reward'] # NOTE: Use unclipped reward 

        # hack to make learning faster (count loss of life as end of episode for memory purposes)
        mem.store(state[0, 0], action, reward, done
                  or (env.ale.lives() != lives))

        state = next_state
        total_reward += reward
        unclipped_reward += info['unclipped_reward']
        frame += 1
        scheduler.step(1)

        if mem.size() < INIT_MEMORY_SIZE:
            continue

        if scheduler.step_count() % UPDATE_FREQ == 0:
            states, next_states, actions, rewards, terminals = mem.sample(
                BATCH_SIZE)
            mask = (1 - terminals).float()
            #y = rewards + mask * GAMMA * torch.max(
            #    target_q_func(next_states), dim=1).values.view(-1, 1).detach()
            # NEW: use average q function
            y = rewards + mask * GAMMA * torch.max(
                avg_q_func(target_q_func_list[-active_target_count:], next_states), dim=1).values.view(-1, 1).detach()
            x = q_func(states)[range(BATCH_SIZE), actions.squeeze()]
            loss = loss_func(x, y.squeeze())
            optimizer.zero_grad()
            loss.backward()

            for param in q_func.parameters():  # gradient clipping
                param.grad.data.clamp_(-1, 1)

            optimizer.step()
            avg_loss += loss.item()
            num_parameter_updates += 1

        if num_parameter_updates % TARGET_UPDATE_EVERY == 0 and scheduler.step_count() % UPDATE_FREQ == 0:  # reset target to source
            #target_q_func.load_state_dict(q_func.state_dict())
            # NEW: Update all target q function with next model
            for idx, target_q_func in enumerate(target_q_func_list):
                if idx != len(target_q_func_list) - 1: # if not last target q function
                    target_q_func.load_state_dict(target_q_func_list[idx+1].state_dict())
                else:
                    target_q_func.load_state_dict(q_func.state_dict())

            if active_target_count < K:
                active_target_count += 1

            print('update target at {}'.format(num_parameter_updates))


    avg_loss /= frame
    avg_reward = 0.9 * avg_reward + 0.1 * total_reward
    avg_unclipped_reward = 0.9 * avg_unclipped_reward + 0.1 * unclipped_reward

    print(
        f"[EPISODE {episode}] Loss: {avg_loss:4f}, " +
        f"Total Reward: {total_reward}, Total Unclipped Reward: {unclipped_reward}, " +
        f"Frames: {frame}, Epsilon: {scheduler.epsilon():4f}, " +
        f"Total Frames: {scheduler.step_count()}, Memory Size: {mem.size()}, " +
        f"Average Q: {avg_q:4f}, " +
        f"Elapsed Time: {int(time.time()-start_time)} sec"
    )

    if episode % PLOT_EVERY == 0:
        with summary_writer.as_default():
            tf.summary.scalar('Total Train Reward', avg_reward, step=scheduler.step_count())
            tf.summary.scalar('Total Unclipped Train Reward', avg_unclipped_reward, step=scheduler.step_count())
            tf.summary.scalar('Epsilon', scheduler.epsilon(), step=scheduler.step_count())
            tf.summary.scalar('Episode Length', frame, step=scheduler.step_count())
            tf.summary.scalar('Average Loss', avg_loss, step=scheduler.step_count())
            tf.summary.scalar('Average Q', avg_q, step=scheduler.step_count())
            tf.summary.scalar('Active Target Count', active_target_count, step=scheduler.step_count())
            tf.summary.scalar('Q', q_values.mean().item(), step=scheduler.step_count())

    if episode % TEST_EVERY == 0 and episode != 0:
        test_reward = test(active_target_count, save=not args.nosave)

    if episode % SAVE_EVERY == 0 and episode != 0 and not args.nosave:
        path = f"episode-{episode}.pt"
        weight_path = os.path.join(weight_dir, path)
        info_path = os.path.join(root_dir, "info.txt")

        torch.save(q_func.state_dict(), weight_path)

        with open(info_path, "a+") as f:
            f.write(",".join([
                str(x) for x in [
                    path,
                    scheduler.step_count(),
                    scheduler.epsilon(), episode, test_reward
                ]
            ]) + "\n")
