from argparse import ArgumentParser
import glob
import os
import time
import numpy as np
import torch
import wandb

from rlpyt.samplers.collections import BatchSpec
from rlpyt.envs.base import  EnvSpaces
from rlpyt.spaces.int_box import IntBox

from src.rlpyt_buffer import AsyncUniformSequenceReplayFrameBufferExtended, SamplesFromReplayExt
from src.algos import ModelSamplesToBuffer, SPRCategoricalDQN
from src.replay_buffer import ReplayBuffer

from src.rlpyt_atari_env import EnvInfo, AtariEnv
from src.agent import AgentInfo, SPRAgent
from src.models import SPRCatDqnModel
from rlpyt.utils.misc import extract_sequences

examples = dict(
    observation = torch.zeros((4,1,84,84)).float(),
    action = torch.ones(1).int() * 0,
    reward = np.array([0.]),
    done = False,
    agent_info = AgentInfo(p=torch.zeros(7)),
    env_info = EnvInfo(game_score=0, traj_done=False),
)

example_to_buffer = ModelSamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            value=examples["agent_info"].p,
        )
replay_kwargs = dict(
            example=example_to_buffer,
            size=1000000,
            B=1,
            batch_T=1,
            discount=0.99,
            n_step_return=10,
            rnn_state_interval=0,
        )

parser = ArgumentParser()
parser.add_argument('--ours', action='store_true')
args = parser.parse_args()

if args.ours:
    rb = ReplayBuffer(**replay_kwargs)
else:
    rb = AsyncUniformSequenceReplayFrameBufferExtended(**replay_kwargs)


algo_kwargs = dict(
    optim_kwargs=dict(
        eps=0.00015,
    ),
    discount=0.99,
    batch_size=32,
    learning_rate=0.0001,
    clip_grad_norm=10.0,
    min_steps_learn=2000,
    double_dqn=True,
    prioritized_replay=0,
    n_step_return=10,
    replay_size=1000000,
    replay_ratio=64,
    target_update_interval=1,
    target_update_tau=1.0,
    eps_steps=2001,
    pri_alpha=0.5,
    pri_beta_steps=100000,
    model_rl_weight=0.0,
    reward_loss_weight=0.0,
    model_spr_weight=5.0,
    t0_spr_loss_weight=0.0,
    time_offset=0,
    distributional=1,
    delta_clip=1.0,
)
agent_kwargs = dict(
    eps_init=1.0,
    eps_final=0.0,
    eps_eval=0.001,
)
model_kwargs = dict(
    dueling=True,
    noisy_nets_std=0.5,
    imagesize=84,
    jumps=0,
    dynamics_blocks=0,
    spr=0,
    noisy_nets=1,
    momentum_encoder=1,
    shared_encoder=0,
    local_spr=0,
    global_spr=1,
    distributional=1,
    renormalize=1,
    norm_type='bn',
    augmentation=['none'],
    q_l1_type=['value', 'advantage'],
    dropout=0.0,
    time_offset=0,
    aug_prob=1.0,
    target_augmentation=0,
    eval_augmentation=0,
    classifier='q_l1',
    final_classifier='linear',
    momentum_tau=0.01,
    dqn_hidden_size=256,
    model_rl=0.0,
    residual_tm=0.0,
)
wandb.init(project="spr vs ours", group="spr")
wandb.define_metric("itr")
wandb.define_metric("tr/*", step_metric="itr")
wandb.define_metric("eps/*", step_metric="itr")

algo = SPRCategoricalDQN(**algo_kwargs)
agent = SPRAgent(ModelCls=SPRCatDqnModel, model_kwargs=model_kwargs, **agent_kwargs)

agent.initialize(
    EnvSpaces(IntBox(low=0,high=255,shape=(4,1,84,84)), IntBox(low=0, high=7)),
    share_memory=False, global_B=1, env_ranks=[0]
)
algo.initialize(
    agent=agent,
    n_itr=10000,
    batch_spec=BatchSpec(B=1, T=1),
    mid_batch_reset=True,
    examples=examples,
    world_size=1,
    rank=0,
)

for fn in sorted(glob.glob(f'../rb/*.pt')):
    data = torch.load(fn)
    samples = ModelSamplesToBuffer(
        observation = data['obs'].unsqueeze(0),
        action = torch.tensor([data['action']]).unsqueeze(0),
        reward = torch.tensor([data['reward'][0]]).unsqueeze(0),
        done = torch.tensor([data['done'][0]]).unsqueeze(0),
        value = torch.ones((1,1,7)) * int(fn.split('/')[-1].split('.')[0]),
    )
    rb.append_samples(samples)

env = AtariEnv(game="assault", episodic_lives=False, seed=0)

# idxes = np.arange(6) + 1
# batch = rb.extract_batch(idxes, np.zeros_like(idxes), 1)
# if args.ours:
#     values = None
# else:
#     values = torch.from_numpy(extract_sequences(rb.samples.value, idxes, np.zeros_like(idxes), 12))
# elapsed_iters = rb.t + rb.T - idxes % rb.T
# elapsed_samples = rb.B * (elapsed_iters)
# batch = SamplesFromReplayExt(*batch, values=values, age=elapsed_samples)

# batch = rb.sample_batch(5)

weights = torch.load('../weights.pt')
nw = {}
for item in weights.items():
    if item[0].split('.')[0] == 'encoder':
        nw.update({'.'.join(item[0].split('.')[1:]):item[1]})
    else:
        nw.update({item[0]:item[1]})

algo.model.load_state_dict(nw)
o = env.reset()

data_log = {
    "eps/rew": 0.,
    "eps/cl_rew": 0.,
    "eps/len": 0,
}

for i in range(10000):
    batch = rb.sample_batch(16)
    loss = algo.loss(batch)[0]

    algo.optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(algo.model.stem_parameters(), algo.clip_grad_norm)
    algo.optimizer.step()

    log_dict = {
        "tr/loss": loss.item(),
        "tr/grad_norm": grad_norm.item(),
        "itr": i
    }

    if i % algo.target_update_interval == 0:
        algo.agent.update_target(algo.target_update_tau)

        # while not os.path.exists(f'../wcheck_{i}.pt'):
        #     time.sleep(1)

        # while True:
        #     try:
        #         tw = torch.load(f'../wcheck_{i}.pt')
        #         os.remove(f'../wcheck_{i}.pt')
        #         print(f"loaded weights {i}")
        #         break
        #     except:
                # pass
        
        # diff = 0.
        # for item in weights.items():
        #     if item[0].split('.')[0] == 'encoder':
        #         diff += (algo.agent.model.state_dict()['.'.join(item[0].split('.')[1:])] - item[1].cpu()).abs().sum()
        #     else:
        #         diff += (algo.agent.model.state_dict()[item[0]] - item[1].cpu()).abs().sum()

        # log_dict.update({"tr/diff": diff})

    out = algo.agent.step(torch.from_numpy(o).unsqueeze(0), None, None)
    env_out = env.step(out.action)

    log_dict.update({"tr/value": out.agent_info.p.mean()})

    o = env_out.observation
    if env_out.env_info.traj_done:
        log_dict.update(data_log)
        log_dict.update({
            "eps/mean_rew": data_log["eps/rew"] / data_log["eps/len"],
            "eps/mean_cl_rew": data_log["eps/cl_rew"] / data_log["eps/len"],
        })

        data_log = {
            "eps/rew": 0.,
            "eps/cl_rew": 0.,
            "eps/len": 0,
        }
        o = env.reset()
    else:
        data_log["eps/rew"] += env_out.env_info.game_score
        data_log["eps/cl_rew"] += env_out.reward
        data_log["eps/len"] += 1

    wandb.log(log_dict)