import glob
import io
import os
import pwd
import torch
from typing import Dict, List, NamedTuple, Tuple
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from multiprocessing import shared_memory, Value

import wandb


def make_shm(base_name: str='', replay_buffer_size: int=1000000, img_size: int=84, channel_dim: int=1) -> List[SharedMemory]:
    try:
        shm = [SharedMemory(name=base_name+'0', create=True, size=replay_buffer_size*(img_size**2)*16*channel_dim),
            SharedMemory(name=base_name+'1', create=True, size=replay_buffer_size*16),
            SharedMemory(name=base_name+'2', create=True, size=replay_buffer_size*16),
            SharedMemory(name=base_name+'3', create=True, size=replay_buffer_size*16),
            SharedMemory(name=base_name+'4', create=True, size=replay_buffer_size*16),]
    except FileExistsError:
        shm = [SharedMemory(name=base_name+str(i)) for i in range(5)]

    return shm


class ReplayBuffer(object):
    def __init__(self, example, size: int=1000000, n_step_return: int=10, 
                 discount: float=0.99, shm: List[shared_memory.SharedMemory]=[], 
                 counter: Value=None, **kwargs) -> None:
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        channel_dim = example.observation.shape[1]
        img_size = example.observation.shape[-1]

        if len(shm) == 0:
            shm = make_shm('spr-shared-mem-rb-', size, img_size, channel_dim)
                           
        if not counter:
            counter = Value('i', 0)

        self._obs_storage = np.ndarray([
            size, channel_dim, img_size, img_size], buffer=shm[0].buf)
        self._action_storage = np.ndarray([size, 1], buffer=shm[1].buf)
        self._reward_storage = np.ndarray([size, 1], buffer=shm[2].buf)
        self._done_storage = np.ndarray([size, 1], buffer=shm[3].buf)
        self._valid_idxs = np.ndarray([size], buffer=shm[4].buf)

        self.shm = shm  # we need to keep references to the shared memory regions around since otherwise the gc collects them and we get segfaults

        self._maxsize = size
        self._next_idx = counter
        
        self.obs_dimensions = channel_dim
        self.n_stacked_frames = example.observation.shape[0]
        self.n_steps_return = n_step_return
        self.gamma = discount

        self.buffer_dir = os.path.join(
            '/checkpoint', pwd.getpwuid(os.getuid())[0], 
            str(os.environ.get('SLURM_JOB_ID')), 'buffer')
        
        os.makedirs(self.buffer_dir, exist_ok=True)

        self.reset_log_dict()

    def __len__(self) -> int:
        return self._next_idx.value

    def reset_log_dict(self):
        self.eps_dict = {
            'eps/ep_rew': 0,
            'eps/ep_len': 0,
            'eps/ep_mean_clipped_rew': 0.,
            'eps/ep_mean_rew': 0.
        }

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def append_samples(self, samples: NamedTuple, traj_infos: List[Dict]) -> None:
        action_probs = samples.value.softmax(-1)
        step_dict = {
           'entropy/entropy': -(action_probs * action_probs.log()).sum(),
           'entropy/max_prob': action_probs.max(),
           'data_step': self._next_idx.value
        }

        self.eps_dict['eps/ep_rew'] += traj_infos[0]['GameScore']
        self.eps_dict['eps/ep_mean_rew'] += traj_infos[0]['GameScore']
        self.eps_dict['eps/ep_mean_clipped_rew'] += samples.reward
        self.eps_dict['eps/ep_len'] += 1

        if any(samples.done):
            self.eps_dict['eps/ep_mean_rew'] /= self.eps_dict['eps/ep_len']
            self.eps_dict['eps/ep_mean_clipped_rew'] /= self.eps_dict['eps/ep_len']
            
            step_dict.update(self.eps_dict)
            self.reset_log_dict()

        wandb.log(step_dict)
        
        self.add(
            samples.observation.flatten(0,2)[-1], # spr gives the obs stacked along the first dimension (we dont want to store the same data mult times)
            samples.action, 
            samples.reward,
            samples.done
        )

    def add(self, obs_t: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done, enable_write: bool=True) -> None:
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param done: (bool) is the episode done
        """
        # This code assumes that there is only a single process writing to the replay buffer
        idx = self._next_idx.value
        self._obs_storage[idx] = obs_t
        self._action_storage[idx] = action
        self._reward_storage[idx] = reward
        self._done_storage[idx] = done
        
        required_steps = self.n_stacked_frames + self.n_steps_return

        if idx > required_steps and not np.any(self._done_storage[idx - required_steps:idx]):
            self._valid_idxs[idx - self.n_steps_return] = 1

        self._next_idx.value = idx + 1

        if enable_write and self._next_idx.value % 256 == 0: # TODO: add to config
            self._write_to_disk(self._next_idx.value - 256, self._next_idx.value)


    def _encode_sample(self, idxes: np.ndarray) -> List:
        observations = np.stack([self._obs_storage[i-self.n_stacked_frames+1:i+1] for i in idxes])
        next_observations = np.stack(
            [self._obs_storage[i+self.n_steps_return-self.n_stacked_frames+1:i+self.n_steps_return+1] for i in idxes])
        rewards = np.array(
            [np.dot(np.power(self.gamma, np.arange(self.n_steps_return)), self._reward_storage[i:i+self.n_steps_return]) for i in idxes])

        return [observations,
                self._action_storage[idxes],
                rewards,
                next_observations,
                self._done_storage[idxes],
                self._valid_idxs[idxes]]

    def sample_batch(self, batch_size: int) -> List:
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = np.random.randint(self.n_stacked_frames, len(self) - 1, size=batch_size)
        return self._encode_sample(idxes)


    def _write_to_disk(self, start_idx: int=0, stop_idx: int=-1) -> None:
        data = (
            self._obs_storage[start_idx:stop_idx],
            self._action_storage[start_idx:stop_idx],
            self._reward_storage[start_idx:stop_idx],
            self._done_storage[start_idx:stop_idx],
            self._valid_idxs[start_idx:stop_idx]
        )
        fn = os.path.join(self.buffer_dir, f"batch_{start_idx:08d}_{stop_idx:08d}.npz")
        save_data(data, fn)

    def restart_from_chpt(self, restart_at_env_step: int=0) -> None:
        for fn in glob.glob(f'{self.buffer_dir}/*.npz'):
            _, env_step, _ = os.path.splitext(os.path.basename(fn))[0].split('_')
            if int(env_step) >= restart_at_env_step:
                os.remove(fn)

        self._next_idx.value = 0

        fns = sorted(glob.glob(f'{self.buffer_dir}/*.npz'))
        for fn in fns:
            obs, actions, rewards, done, valid = load_data(fn)
            
            _, env_step, _ = os.path.splitext(os.path.basename(fn))[0].split('_')
            env_step = int(env_step)
            for idx in range(int(env_step), min(int(env_step) + len(obs), restart_at_env_step)):
                assert idx == self._next_idx.value

                self.add(obs[idx-env_step], actions[idx-env_step], rewards[idx-env_step], done[idx-env_step], False)

            if int(env_step) + len(obs) >= restart_at_env_step:
                os.remove(fn)
                
        self._done_storage[self._next_idx.value - 1] = True

    def close(self) -> None:
        """
        Closes access to the shared memory from this instance. In order to ensure
        proper cleanup of resources, all instances should call close() once the 
        instance is no longer needed. Note that calling close() does not cause the 
        shared memory block itself to be destroyed.
        """
        for shmi in self.shm:
            shmi.close()

    def unlink(self) -> None:
        """
        Requests that the underlying shared memory block be destroyed.
        """
        for shmi in self.shm:
            shmi.unlink()
        

def save_data(data, fn: str) -> None:
    with io.BytesIO() as bs:
        np.savez_compressed(bs, *data)
        bs.seek(0)
        with open(fn, 'wb') as f:
            f.write(bs.read())

def load_data(fn: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(fn, 'rb') as f:
        episode = np.load(f)
        return episode['arr_0.npy'], episode['arr_1.npy'], episode['arr_2.npy'], episode['arr_3.npy'], episode['arr_4.npy']
