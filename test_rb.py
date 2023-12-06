import numpy as np
import torch
import wandb

from rlpyt.samplers.collections import BatchSpec
from rlpyt.envs.base import  EnvSpaces
from rlpyt.spaces.int_box import IntBox

from src.rlpyt_buffer import AsyncUniformSequenceReplayFrameBufferExtended, SamplesFromReplayExt
from src.algos import ModelSamplesToBuffer, SPRCategoricalDQN

from src.rlpyt_atari_env import EnvInfo
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
wandb.init()
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

for i in range(100):
    samples = ModelSamplesToBuffer(
        observation = torch.ones((1,1,4,1,84,84)).float() * i,
        action = torch.ones((1,1)).int() * (i % 7),
        reward = torch.ones((1,1)) * i,
        done = torch.ones((1,1)) * (i == 5),
        value = torch.ones((1,1,7)) * i,
    )
    rb.append_samples(samples)

# rb.sample_batch(10)

idxes = np.arange(6) + 1
batch = rb.extract_batch(idxes, np.zeros_like(idxes), 1)
values = torch.from_numpy(extract_sequences(rb.samples.value, idxes, np.zeros_like(idxes), 12))
elapsed_iters = rb.t + rb.T - idxes % rb.T
elapsed_samples = rb.B * (elapsed_iters)
batch = SamplesFromReplayExt(*batch, values=values, age=elapsed_samples)

# batch = rb.sample_batch(5)

weights = torch.load('../weights.pt')
nw = {}
for item in weights.items():
    if item[0].split('.')[0] == 'encoder':
        nw.update({'.'.join(item[0].split('.')[1:]):item[1]})
    else:
        nw.update({item[0]:item[1]})

algo.model.load_state_dict(nw)

loss = algo.loss(batch)

batch = rb.sample_batch(5)
# batch = SamplesFromReplayExt(*batch, values=)