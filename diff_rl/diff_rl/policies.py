from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from diff_rl.common.helpers import SinusoidalPosEmb
from diff_rl.common.policies import ContinuousCritic

class Actor(BasePolicy): # get x_t + sigma_xt, output x_0

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=False,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        
        # here generate actor net for specifying the structure of self.mu
        action_dim = get_action_dim(self.action_space)
        self.mu =  MLP(state_dim=self.features_dim, action_dim=action_dim)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, x, time, obs):
        if obs.dim()==3:
            features = obs.float()
        else:
            features = self.extract_features(obs, self.features_extractor) # flatten
        return self.mu(x, time, features)
    
    def _predict(self, observation, deterministic = False):
        action = self(observation)
        return action


class Corrector(BasePolicy): # get x_t, output corrected bias sigma_xt

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=False,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        
        # here generate actor net for specifying the structure of self.mu
        action_dim = get_action_dim(self.action_space)
        self.mu =  Corrector_MLP(state_dim=self.features_dim, action_dim=action_dim)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, x, time, obs):
        if obs.dim()==3:
            features = obs.float()
        else:
            features = self.extract_features(obs, self.features_extractor) # flatten
        return self.mu(x, time, features)
    
    def _predict(self, observation, deterministic = False):
        action = self(observation)
        return action

    
class TD3Policy(BasePolicy):

    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture, from the original paper
        net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.corrector_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule): # TODO, add corrector here
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        
        self.corrector = self.make_corrector(features_extractor=None)
        self.corrector_target = self.make_corrector(features_extractor=None)
        self.corrector_target.load_state_dict(self.corrector.state_dict())

        self.corrector.optimizer = self.optimizer_class(
            self.corrector.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)
        self.corrector_target.set_training_mode(False)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor = None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor = None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)
    
    def make_corrector(self, features_extractor = None):
        corrector_kwargs = self._update_features_extractor(self.corrector_kwargs, features_extractor)
        return Corrector(**corrector_kwargs).to(self.device)

    def forward(self, observation, deterministic: bool = False):
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation, deterministic: bool = False):
        action = self.actor(observation)
        return action

    def set_training_mode(self, mode):
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

MlpPolicy = TD3Policy

class MLP(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 t_dim=16):

        super(MLP, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.layers = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, action_dim),
                                       nn.Tanh()) 

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        if state.dim() == 3: # TODO, be careful here, cuz it can only repeat 50 times
            t = th.repeat_interleave(t.unsqueeze(1), repeats=50, dim=1)
        if state is not None:
            x = th.cat([x, t, state], dim=-1)
        else:
            x = th.cat([x, t], dim=1)
        
        x = self.layers(x)
        return x
    
class Corrector_MLP(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 t_dim=16):

        super(Corrector_MLP, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.layers = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, action_dim)) 

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        if state.dim() == 3: # TODO, be careful here, cuz it can only repeat 50 times
            t = th.repeat_interleave(t.unsqueeze(1), repeats=50, dim=1)
        if state is not None:
            x = th.cat([x, t, state], dim=-1)
        else:
            x = th.cat([x, t], dim=1)
        
        x = self.layers(x)
        return x