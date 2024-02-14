import glob
import io
import os
import time
import torch
from typing import List, Tuple

import numpy as np
from multiprocessing import shared_memory, Value

from rlpyt.replays.sequence.n_step import SamplesFromReplay
import wandb


def make_shm(base_name: str='', replay_buffer_size: int=1000000, img_size: int=84, channel_dim: int=1) -> List[shared_memory.SharedMemory]:
    try:
        shm = [shared_memory.SharedMemory(name=base_name+'0', create=True, size=replay_buffer_size*(img_size**2)*16*channel_dim),
            shared_memory.SharedMemory(name=base_name+'1', create=True, size=replay_buffer_size*16),
            shared_memory.SharedMemory(name=base_name+'2', create=True, size=replay_buffer_size*16),
            shared_memory.SharedMemory(name=base_name+'3', create=True, size=replay_buffer_size*16),
            shared_memory.SharedMemory(name=base_name+'4', create=True, size=replay_buffer_size*16),]
    except FileExistsError:
        shm = [shared_memory.SharedMemory(name=base_name+str(i)) for i in range(5)]

    return shm


class ReplayBuffer(object):
    def __init__(self, 
                 example, size, B, batch_T, discount, n_step_return,
                 counter: Value=None,
                 shm: List[shared_memory.SharedMemory]=None, **kwargs) -> None:
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        if counter is None:
            counter = Value('i', 0)
        # if shm is None:
        #     shm = make_shm("spr_shared_mem_" + str(int(time.time())), img_size=example.observation.shape[-1], channel_dim=example.observation.shape[1])
        self._obs_storage = np.ndarray([
            size + example.observation.shape[0] - 1, 1, *example.observation.shape[1:]], dtype=float)
        self._action_storage = np.ndarray([size, 1])
        self._reward_storage = np.ndarray([size, 1])
        self._done_storage = np.ndarray([size, 1])
        self._valid_idxs = np.ndarray([size])

        self.shm = shm  # we need to keep references to the shared memory regions around since otherwise the gc collects them and we get segfaults

        self.n_stacked = example.observation.shape[-4]
        self.n_step = n_step_return
        self.gamma = discount

        self._maxsize = size
        self._next_idx = counter

        # self.buffer_dir = config.buffer_dir
        

    def __len__(self) -> int:
        return self._next_idx.value

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) - self.n_step >= n_samples

    def append_samples(self, samples):
        self.add(
            samples.observation[:,:,-1],
            samples.action, samples.reward, samples.done,
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
        self._obs_storage[[idx + self.n_stacked - 1]] = obs_t
        self._action_storage[idx] = action
        self._reward_storage[idx] = reward
        self._done_storage[idx] = done

        self._next_idx.value = idx + 1

        if enable_write and self._next_idx.value % 256 == 0: # TODO: add to config
            self._write_to_disk(self._next_idx.value - 256, self._next_idx.value)


    def extract_batch(self, idxes: np.ndarray, *args) -> List: # TODO: check indexes
        returns = []
        reward = []
        actions = []
        done_n = []

        gamma_support = np.power(self.gamma, np.arange(self.n_step))

        for i in idxes:
            multiplier = np.expand_dims((1 - self._done_storage[i:i+self.n_step].cumsum()) + self._done_storage[i:i+self.n_step].squeeze(-1), -1)
            r_i = np.dot(gamma_support, self._reward_storage[i:i+self.n_step] * multiplier)
            returns.append(r_i)

            if i == 0:
                actions.append(np.concatenate([self._action_storage[i-1:], self._action_storage[:i+self.n_step]]))
                reward.append(np.concatenate([self._reward_storage[i-1:], self._reward_storage[:i+self.n_step]]))
            else:
                actions.append(self._action_storage[i-1:i+self.n_step])
                reward.append(self._reward_storage[i-1:i+self.n_step])

            done_n.append(np.any(self._done_storage[i:i+self.n_step]))

        observations = self.extract_observation(idxes, np.zeros_like(idxes), 1 + self.n_step)

        done_n = np.stack(done_n)

        return SamplesFromReplay(
            all_observation = torch.from_numpy(observations).to(torch.uint8),
            all_action = torch.from_numpy(np.stack(actions, 1)).int(),
            all_reward = torch.from_numpy(np.stack(reward, 1)),
            return_ = torch.from_numpy(np.stack(returns)).unsqueeze(0),
            done = torch.from_numpy(self._done_storage[idxes]).T,
            done_n = torch.from_numpy(done_n).unsqueeze(0),
            init_rnn_state = None
        )

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
        idxes = np.random.randint(0, len(self) - self.n_step - self.n_stacked, size=batch_size)
        wandb.log({'mean_sampled_idx': idxes.mean()})
        return self.extract_batch(idxes)


    def _write_to_disk(self, start_idx: int=0, stop_idx: int=-1) -> None:
        data = (
            self._obs_storage[start_idx:stop_idx],
            self._action_storage[start_idx:stop_idx],
            self._reward_storage[start_idx:stop_idx],
            self._done_storage[start_idx:stop_idx],
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
            obs, actions, rewards, done = load_data(fn)
            
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

    def extract_observation(self, T_idxs, B_idxs, T):
        """Observations are re-assembled from frame-wise buffer as [T,B,C,H,W],
        where C is the frame-history channels, which will have redundancy across the
        T dimension.  Frames are returned OLDEST to NEWEST along the C dimension.

        Frames are zero-ed after environment resets."""
        observation = np.empty(shape=(T, len(B_idxs), self.n_stacked) +  # [T,B,C,H,W]
            self._obs_storage.shape[2:], dtype=self._obs_storage.dtype)
        fm1 = self.n_stacked - 1
        for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
            if t + T > self._maxsize:  # wrap (n_stacked duplicated)
                m = self._maxsize - t
                w = T - m
                for f in range(self.n_stacked):
                    observation[:m, i, f] = self._obs_storage[t + f:t + f + m, b]
                    observation[m:, i, f] = self._obs_storage[f:w + f, b]
            else:
                for f in range(self.n_stacked):
                    observation[:, i, f] = self._obs_storage[t + f:t + f + T, b]

            # Populate empty (zero) frames after environment done.
            if t - fm1 < 0 or t + T > self._maxsize:  # Wrap.
                done_idxs = np.arange(t - fm1, t + T) % self._maxsize
            else:
                done_idxs = slice(t - fm1, t + T)
            done_fm1 = self._done_storage[done_idxs, b]
            if np.any(done_fm1):
                where_done_t = np.where(done_fm1)[0] - fm1  # Might be negative...
                for f in range(1, self.n_stacked):
                    t_blanks = where_done_t + f  # ...might be > T...
                    t_blanks = t_blanks[(t_blanks >= 0) & (t_blanks < T)]  # ..don't let it wrap.
                    observation[t_blanks, i, :self.n_stacked - f] = 0

        return observation
        

def save_data(data, fn: str) -> None:
    with io.BytesIO() as bs:
        np.savez_compressed(bs, *data)
        bs.seek(0)
        with open(fn, 'wb') as f:
            f.write(bs.read())

def load_data(fn: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(fn, 'rb') as f:
        episode = np.load(f)
        return episode['arr_0.npy'], episode['arr_1.npy'], episode['arr_2.npy'], episode['arr_3.npy'] #, episode['arr_4.npy']
