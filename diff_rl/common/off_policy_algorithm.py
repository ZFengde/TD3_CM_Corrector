import io
import pathlib
import sys
import time
import warnings
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from diff_rl.common.buffers import DictReplayBuffer, ReplayBuffer
from diff_rl.common.consistency import Consistency_Model


class OffPolicyAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}
        self._episode_storage = None

        # Save train freq parameter, will be converted later to TrainFreq object
        self.train_freq = train_freq

        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        self.use_sde_at_warmup = use_sde_at_warmup

    def _convert_train_freq(self):
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))  # type: ignore[assignment]
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)  # type: ignore[assignment,arg-type]

    def _setup_model(self): # this is where to build the models
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self.consistency_model = Consistency_Model(env=self.env, 
                                                    sigma_max=80.0,
                                                    device=self.device,
                                                    sampler="onestep",)
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]):
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ):
        
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.replay_buffer.set_env(self.env)
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()

        # Update saved replay buffer device to match current setting, see GH#1561
        self.replay_buffer.device = self.device

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ):

        replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            assert replay_buffer is not None  # for mypy
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        assert self.env is not None, "You must set the environment before calling _setup_learn()"
        # Vectorize action noise if needed
        if (
            self.action_noise is not None
            and self.env.num_envs > 1
            and not isinstance(self.action_noise, VectorizedActionNoise)
        ):
            self.action_noise = VectorizedActionNoise(self.action_noise, self.env.num_envs)

        return super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int):

        raise NotImplementedError()

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ):
        # Select action randomly or according to policy # Always output scaled actions here
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)]) # this is action among low and high
            scaled_action = self.policy.unscale_action(unscaled_action)
            # this is unscale actions
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"

            state = th.FloatTensor(self._last_obs.reshape(1, -1)).to(self.device)
            state_rpt = th.repeat_interleave(state, repeats=50, dim=0)

            # This is sample from consistency model
            scaled_action = self.consistency_model.sample(model=self.actor, corrector=self.corrector, state=state_rpt)
            q_value = self.critic_target.q1_forward(state_rpt, scaled_action).flatten()
            soft_q_value = F.softmax(q_value, dim=0)
            idx = th.multinomial(soft_q_value, num_samples=1) # this mechanism works a bit since all q-values are similar
            scaled_action = scaled_action[idx].cpu().data.numpy()

            # unscaled_action is the direct output from the model
            
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action) # action from [low, high]
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action # true action is among low and high but buffer_action is among(-1, 1)

    def _dump_logs(self):
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self):
        pass

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ):
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ):
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            # actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones) # only when episode done, 

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def save(self, models_dir, training_step):
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                print(f'logging to {models_dir}/{training_step}')
            th.save(self.policy.state_dict(), f'{models_dir}/{training_step}') 

    def load(self, path):
        self.policy.load_state_dict(th.load(path))