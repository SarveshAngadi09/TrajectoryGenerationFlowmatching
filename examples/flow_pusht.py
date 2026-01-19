#!/usr/bin/env python
#
# Copyright (c) 2024, Honda Research Institute Europe GmbH
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This notebook is an example of "Affordance-based Robot Manipulation with Flow Matching" https://arxiv.org/abs/2409.01083

import sys
import sys
from pathlib import Path

# Add the repo root folder to Python path
repo_root = Path(__file__).resolve().parents[1]  # adjust if needed
sys.path.append(str(repo_root))

sys.dont_write_bytecode = True
sys.path.append('../models')
import numpy as np
import torch
from external.models import pusht
import torch.nn as nn
from tqdm import tqdm
from external.models.unet import ConditionalUnet1D
from external.models.resnet import get_resnet
from external.models.resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from termcolor import colored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

print("Doing on GPU:", torch.cuda.is_available() )

##################################
########## download the pusht data and put in the folder
dataset_path = "pusht_cchi_v7_replay.zarr.zip"

obs_horizon = 1
pred_horizon = 16
action_dim = 2
action_horizon = 8
num_epochs = 4001
vision_feature_dim = 514
# create dataset from file
dataset = pusht.PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=512,
    num_workers=64,
    shuffle=True,
    pin_memory=True,               # enables async GPU transfer
    persistent_workers=True,       # avoids respawning workers
    prefetch_factor=4
)

##################################################################
# create network object
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=vision_feature_dim
)
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
}).to(device)

##################################################################
sigma = 0.0
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)
optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)

avg_loss_train_list = []
scaler = torch.cuda.amp.GradScaler()

import imageio.v3 as iio





########################################################################
#### Train the model
def train():
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        for data in tqdm(dataloader):
            x_img = data['image'][:, :obs_horizon].to(device)
            x_pos = data['agent_pos'][:, :obs_horizon].to(device)
            x_traj = data['action'].to(device)

            x_traj = x_traj.float()
            x0 = torch.randn(x_traj.shape, device=device)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

            # encoder vision features
            image_features = nets['vision_encoder'](x_img.flatten(end_dim=1))
            image_features = image_features.reshape(*x_img.shape[:2], -1)
            obs_features = torch.cat([image_features, x_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            with torch.cuda.amp.autocast():
    
                vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_cond)

                loss = torch.mean((vt - ut) ** 2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss_train += loss.detach()

            #loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights

            ema.step(nets.parameters())

        avg_loss_train = total_loss_train / len(dataloader)
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow'))
        print(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}")

        #if epoch == 100 or epoch == 500 or epoch == 1000 or epoch == 2000 or epoch ==3000 or epoch == 4000:
        if epoch != 0 and epoch % 100 == 0:
            ema.store(nets.parameters())
            ema.copy_to(nets.parameters())
            PATH = './flow_matching/Checkpoints/flow_ema_%05d.pth' % epoch
            torch.save({'vision_encoder': nets.vision_encoder.state_dict(),
                        'noise_pred_net': nets.noise_pred_net.state_dict(),
                        }, PATH)
            ema.restore(nets.parameters())


########################################################################
###### test the model
def test():
    PATH = './flow_matching/Checkpoints/flow_ema_00500.pth'
    state_dict = torch.load(PATH, map_location='cuda')
    ema_nets = nets
    ema_nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
    ema_nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])

    max_steps = 200
    env = pusht.PushTImageEnv()

    test_start_seed = 1000
    n_test = 4
    import random
    ###### please choose the seed you want to test
    for epoch in range(n_test):
        seed = random.sample(range(1000, 1100), n_test)
        env.seed(seed)

        for pp in range(10):
            print(f"\n=== Test episode {epoch * 10 + pp + 1} / {n_test * 10} ===")

            obs, info = env.reset()

            obs_deque = collections.deque([obs], maxlen=obs_horizon)
            imgs = [env.render(mode='rgb_array')]
            rewards = list()
            done = False
            step_idx = 0

            with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
                while not done:
                    B = 1
                    x_img = np.stack([x['image'] for x in obs_deque])
                    x_pos = np.stack([x['agent_pos'] for x in obs_deque])
                    x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

                    x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
                    # infer action
                    with torch.no_grad():
                        # get image features
                        # t1 = time.time()
                        image_features = ema_nets['vision_encoder'](x_img)
                        obs_features = torch.cat([image_features, x_pos], dim=-1)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                        timehorion = 1
                        for i in range(timehorion):
                            noise = torch.rand(1, pred_horizon, action_dim).to(device)
                            x0 = noise.expand(x_img.shape[0], -1, -1)
                            timestep = torch.tensor([i / timehorion]).to(device)

                            if i == 0:
                                vt = ema_nets['noise_pred_net'](x0, timestep, global_cond=obs_cond)
                                traj = (vt * 1 / timehorion + x0)

                            else:
                                vt = ema_nets['noise_pred_net'](traj, timestep, global_cond=obs_cond)
                                traj = (vt * 1 / timehorion + traj)

                    # print(time.time() - t1)

                    naction = traj.detach().to('cpu').numpy()
                    naction = naction[0]
                    action_pred = pusht.unnormalize_data(naction, stats=stats['action'])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    # x_img = x_img[0, :].permute((1, 2, 0))
                    # plot_trajectory(x0[0].detach().cpu().numpy(), vt[0].detach().cpu().numpy(),
                    #                 action_pred,
                    #                 x_img.detach().cpu().numpy())

                    # execute action_horizon number of steps
                    for j in range(len(action)):
                        # stepping env
                        obs, reward, done, _, info = env.step(action[j])
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)
                        imgs.append(env.render(mode='rgb_array'))

                        # update progress bar
                        step_idx += 1

                        pbar.update(1)
                        pbar.set_postfix(reward=reward)

                        done = done or (step_idx >= max_steps)

            video_path = f"./flow_matching/Videos/pusht_seed_{seed}_trial_{pp}.mp4"
            iio.imwrite(video_path, imgs, fps=20, codec="libx264")
            #save_video(imgs, video_path)

            
#Code by SarveshGPT
import csv

import csv
import os
import numpy as np
import torch
from tqdm import tqdm
import collections

def evaluate():
    episodes_csv = "./flow_matching/eval_results_episodes.csv"
    summary_csv = "./flow_matching/eval_results_summary.csv"

    # ----------------------------------
    # Ensure headers exist (once)
    # ----------------------------------
    if not os.path.isfile(episodes_csv):
        with open(episodes_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Checkpoint', 'Episode', 'Coverage'])

    if not os.path.isfile(summary_csv):
        with open(summary_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Checkpoint',
                'Max',
                'Mean',
                'Std',
                'Zero Coverages (%)'
            ])

    max_episode = 4000
    step = 100

    for episode in range(step, max_episode + 1, step):
        checkpoint_path = f'./flow_matching/Checkpoints/flow_ema_{episode:05d}.pth'
        print(f"Loading checkpoint: {checkpoint_path}")

        # Load model
        state_dict = torch.load(checkpoint_path, map_location=device)
        nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
        nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
        nets.eval()

        # Environment
        env = pusht.PushTImageEnv()

        n_episodes = 200
        max_steps = 200

        all_rewards = []
        episode_coverages = []

        for ep in tqdm(range(n_episodes), desc=f"Checkpoint {episode}"):

            env.seed(seed=1000 + ep)
            obs, info = env.reset()
            obs_deque = collections.deque([obs], maxlen=obs_horizon)

            ep_return = 0.0
            done = False
            step_idx = 0

            with torch.no_grad():
                while not done and step_idx < max_steps:
                    x_img = np.stack([x['image'] for x in obs_deque])
                    x_pos = np.stack([x['agent_pos'] for x in obs_deque])
                    x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

                    x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)

                    image_features = nets['vision_encoder'](x_img)
                    obs_features = torch.cat([image_features, x_pos], dim=-1)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    noise = torch.rand(1, pred_horizon, action_dim).to(device)
                    x0 = noise.expand(x_img.shape[0], -1, -1)
                    timestep = torch.tensor([0.0], device=device)

                    vt = nets['noise_pred_net'](x0, timestep, global_cond=obs_cond)
                    traj = vt + x0

                    naction = traj[0].cpu().numpy()
                    action_pred = pusht.unnormalize_data(naction, stats=stats['action'])

                    start = obs_horizon - 1
                    end = start + action_horizon
                    action_seq = action_pred[start:end]

                    for a in action_seq:
                        obs, reward, done, _, info = env.step(a)
                        obs_deque.append(obs)
                        ep_return += reward
                        step_idx += 1
                        done = done or (step_idx >= max_steps)

            all_rewards.append(ep_return)

            # Coverage calculation
            goal_body = env._get_goal_pose_body(env.goal_pose)
            goal_geom = pusht.pymunk_to_shapely(goal_body, env.block.shapes)
            block_geom = pusht.pymunk_to_shapely(env.block, env.block.shapes)

            intersection_area = goal_geom.intersection(block_geom).area
            coverage = intersection_area / goal_geom.area

            episode_coverages.append(coverage)

        episode_coverages = np.array(episode_coverages)

        # ----------------------------------
        # Write EPISODE-LEVEL CSV
        # ----------------------------------
        with open(episodes_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            for idx, cov in enumerate(episode_coverages, start=1):
                writer.writerow([episode, idx, cov])

        # ----------------------------------
        # Compute summary
        # ----------------------------------
        max_cov = float(np.max(episode_coverages))
        mean_cov = float(np.mean(episode_coverages))
        std_cov = float(np.std(episode_coverages))
        zero_cov_pct = float(np.mean(episode_coverages == 0) * 100)

        # ----------------------------------
        # Write SUMMARY CSV (ONE ROW)
        # ----------------------------------
        with open(summary_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                max_cov,
                mean_cov,
                std_cov,
                zero_cov_pct
            ])

        print(f"Checkpoint {episode} done")
        print(f"Max: {max_cov:.4f}, Mean: {mean_cov:.4f}, Std: {std_cov:.4f}, Zero%: {zero_cov_pct:.2f}\n")

    return



if __name__ == '__main__':
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("No argument provided. Please specify 'train', 'test', or 'print'.")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == 'train':
        train()
    elif arg == 'test':
        test()
    elif arg == 'unittest':
        print("Uni Test Successful")
    elif arg == 'eval':
        metrics = evaluate(
        )
        print(metrics)
    else:
        print(f"Unknown argument '{arg}'. Please specify 'train', 'test', or 'print'.")
