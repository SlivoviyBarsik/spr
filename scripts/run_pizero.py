from collections import deque
import torch.multiprocessing as mp

from gym.vector.utils import CloudpickleWrapper
from src.async_mcts import AsyncMCTS
from src.mcts_memory import LocalBuffer, initialize_replay_buffer, samples_to_buffer
from src.model_trainer import TrainingWorker
from src.async_reanalyze import AsyncReanalyze
from src.utils import get_args
from src.logging import log_results

import os
import copy
import torch
import wandb
import numpy as np
import gym

from src.vectorized_mcts import VectorizedMCTS


def run_pizero(args):
    buffer = initialize_replay_buffer(args)
    devices = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()
    send_queue = ctx.Queue()
    receive_queues = []
    for i in range(devices):
        receive_queue = mp.Queue()
        worker = TrainingWorker(i,
                                devices,
                                devices,
                                args,
                                send_queue,
                                error_queue,
                                receive_queue,
                                CloudpickleWrapper(buffer))
        process = ctx.Process(target=worker.optimize, args=())
        process.start()
        receive_queues.append(receive_queue)

    # Need to get the target network from the training agent that created it.
    network = send_queue.get(block=True, timeout=None)
    if args.target_update_interval > 0:
        target_network = copy.deepcopy(network)
    else:
        target_network = network
    target_network.share_memory()

    if args.reanalyze:
        async_reanalyze = AsyncReanalyze(args, target_network, debug=args.debug_reanalyze)

    local_buf = LocalBuffer()
    env_steps = 0
    eprets = np.zeros(args.num_envs, 'f')
    episode_rewards = deque(maxlen=10)
    wandb.log({'env_steps': 0})

    env = gym.vector.make('atari-v0', num_envs=args.num_envs, args=args,
                          asynchronous=not args.sync_envs)

    # TODO return int observations
    obs = env.reset()
    obs = torch.from_numpy(obs)
    vectorized_mcts = VectorizedMCTS(args, env.action_space[0].n, args.num_envs, target_network)
    eval_vectorized_mcts = VectorizedMCTS(args, env.action_space[0].n, args.evaluation_episodes, target_network, eval=True)
    total_episodes = 0.
    total_train_steps = 0
    while env_steps < args.total_env_steps:
        # Run MCTS for the vectorized observation
        actions, policies, values = vectorized_mcts.run(obs)
        policy_probs = policies.probs
        next_obs, reward, done, infos = env.step(actions.cpu().numpy())
        reward, done = torch.from_numpy(reward).float(), torch.from_numpy(done).float()
        obs, actions, reward, done, policy_probs, values = obs.cpu(), actions.cpu(), reward.cpu(),\
                                                           done.cpu(), policy_probs.cpu(), values.cpu()

        eprets += np.array(reward)

        for i in range(args.num_envs):
            if done[i]:
                episode_rewards.append(eprets[i])
                wandb.log({'Episode Reward': eprets[i],
                           'env_steps': env_steps})
                eprets[i] = 0

        if args.reanalyze:
            async_reanalyze.store_transitions(
                obs[:, -1],
                actions,
                reward,
                done,
            )

            total_episodes += torch.sum(done)

            # Add the reanalyzed transitions to the real data.
            new_samples = async_reanalyze.get_transitions(total_episodes)
            cat_obs = torch.cat([obs, new_samples[0]], 0)
            actions = torch.cat([actions, new_samples[1]], 0)
            reward = torch.cat([reward, new_samples[2]], 0)
            done = torch.cat([done, new_samples[3]], 0)
            policy_probs = torch.cat([policy_probs, new_samples[4]], 0)
            values = torch.cat([values, new_samples[5]], 0)
            local_buf.append(cat_obs, actions, reward, done, policy_probs, values)

        else:
            local_buf.append(obs, actions, reward, done, policy_probs, values)

        if env_steps % args.jumps == 0 and env_steps > 0:
            # Send transitions from the local buffer to the replay buffer
            buffer.append_samples(samples_to_buffer(*local_buf.stack()))
            local_buf.clear()

        if not send_queue.empty():
            steps, log = send_queue.get()
            log_results(log, steps)
            target_train_steps = steps

            if (args.target_update_interval > 0 and
                (total_train_steps % args.target_update_interval >
                 target_train_steps % args.target_update_interval or
                 target_train_steps - total_train_steps >=
                 args.target_update_interval)):

                print("Updated target weights at step {}".format(target_train_steps))
                target_network.load_state_dict(network.state_dict())
            total_train_steps = target_train_steps

        # Send a command to start training if ready
        if args.env_steps*101 >= env_steps > args.num_envs*100:
            [q.put("train") for q in receive_queues]

        if env_steps % args.log_interval == 0 and len(episode_rewards) > 0:
            print('Env Steps: {}, Mean Reward: {}, Median Reward: {}'.format(env_steps, np.mean(episode_rewards),
                                                                             np.median(episode_rewards)))
            wandb.log({'Mean Reward': np.mean(episode_rewards), 'Median Reward': np.median(episode_rewards),
                       'env_steps': env_steps})

        if env_steps % args.evaluation_interval == 0 and not args.debug_reanalyze:
            avg_reward = eval_vectorized_mcts.evaluate()
            print('Env steps: {}, Avg_Reward: {}'.format(env_steps, avg_reward))
            wandb.log({'env_steps': env_steps, 'avg_reward': avg_reward})

        obs.copy_(torch.from_numpy(next_obs))
        env_steps += args.num_envs


if __name__ == '__main__':
    args = get_args()
    tags = []
    try:
        if len(args.name) == 0:
            run = wandb.init(project=args.wandb_proj,
                             entity="abs-world-models",
                             tags=tags,
                             config=vars(args))
        else:
            run = wandb.init(project=args.wandb_proj,
                             name=args.name,
                             entity="abs-world-models",
                             tags=tags,
                             config=vars(args))
    except wandb.run_manager.LaunchError as e:
        print(e)
        pass

    if len(args.savedir) == 0:
        args.savedir = os.environ["SLURM_TMPDIR"]
    args.savedir = "{}/{}".format(args.savedir, run.id)
    os.makedirs(args.savedir, exist_ok=True)

    print("Saving episode data in {}".format(args.savedir))

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
    else:
        args.device = torch.device('cpu')

    run_pizero(args)
