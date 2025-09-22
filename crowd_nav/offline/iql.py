# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import h5py
import argparse
import configparser
import logging

import torchvision
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from st import ST
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from crowd_sim.envs.utils.info import *
from crowd_nav.reward_model import TransformerRewardModel
from crowd_nav.utils import reward_from_preference

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cpu"
    env: str = "CrowdSim-v0"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(500)  # How often (time steps) we evaluate
    n_episodes: int = 20  # How many episodes run during evaluation
    max_timesteps: int = int(100000)  # Max time steps to run environment
    checkpoints_path: Optional[str] = ""  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 1000000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "IQL"
    reward_model_path: str = os.path.join(project_root,
                                          "reward_model_logs/CrowdSim-v0/transformer/epoch_10_query_2000_len_15_seed_0/models",
                                          "reward_model.pt")
    reward_model_type: str = "transformer"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        # env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


cnt = 20000


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, list1: List, list2: List, list3: List, list4: List, device: str,
        n_episodes: int, seed: int
):
    global cnt
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default=os.path.join(project_root, "configs", "env.config"))
    args = parser.parse_args()
    actor.eval()
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(actor)
    # robot.print_info()
    phase = 'train'
    success = 0
    collision = 0
    run_time = 0
    avg_nav_time = 0
    avg_reward = 0
    for _ in range(n_episodes):
        observation = env.reset(phase=phase, test_case=cnt)
        done = False
        rewards = 0
        gamma = 1  # 折扣因子
        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state = to_np(transform(joint_state, device).view(1, -1).squeeze(0))
            # print(state.shape)  # 输出为（65，）
            action = actor.act(state)
            # print(action.shape)  # 输出为（2，）
            action = ActionXY(action[0], action[1])
            observation, reward, done, info = env.step(action)
            # print(reward)
            rewards += reward * gamma
            gamma = gamma * 0.99
        if isinstance(info, ReachGoal):
            success += 1
            avg_nav_time += env.global_time
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            run_time += 1

        avg_reward += rewards
        cnt += 1
    if success != 0:
        avg_nav_time = avg_nav_time / success
    else:
        avg_nav_time = 0
    avg_reward = avg_reward / n_episodes
    print(
        f"成功次数为{success} 成功率为{success / n_episodes} 碰撞次数{collision} 超时次数{run_time} 平均导航时间为{avg_nav_time} 平均奖励为{avg_reward} ")
    list1.append(success / n_episodes)
    list2.append(collision / n_episodes)
    list3.append(avg_nav_time)
    list4.append(avg_reward)
    actor.train()


@torch.no_grad()
def test_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
):
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default=os.path.join(project_root, "configs", "env.config"))
    args = parser.parse_args()

    actor.eval()
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(actor)
    # robot.print_info()
    phase = 'test'
    success = 0
    collision = 0
    run_time = 0
    avg_nav_time = 0
    avg_reward = 0
    for _ in range(n_episodes):
        observation, done = env.reset(phase=phase), False
        rewards = 0
        gamma = 1  # 折扣因子
        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state = to_np(transform(joint_state, device).view(1, -1).squeeze(0))
            # print(state.shape)  # 输出为（65，）
            action = actor.act(state)
            # print(action.shape)  # 输出为（2，）
            action = ActionXY(action[0], action[1])
            observation, reward, done, info = env.step(action)
            # print(reward)
            rewards += reward * gamma
            gamma = gamma * 0.99
        if isinstance(info, ReachGoal):
            success += 1
            avg_nav_time += env.global_time
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            run_time += 1
        # print(env.global_time)

        avg_reward += rewards
    avg_nav_time = avg_nav_time / success
    avg_reward = avg_reward / n_episodes
    print(
        f"成功次数为{success} 成功率为{success / n_episodes} 碰撞次数{collision} 超时次数{run_time} 平均导航时间为{avg_nav_time} 平均奖励为{avg_reward} ")


def transform(state, device):
    state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(device)
                              for human_state in state.human_states], dim=0)

    state_tensor = rotate(state_tensor)
    return state_tensor


def rotate(state):
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    theta = torch.zeros_like(v_pref)
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1
    da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                              reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
    # dg = ||p − pg||2 is the robot’s distance to the goal and di = ||p − pi||2 is the robot’s distance to the neighbor i
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [128, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        # self.attention = SelfAttention()
        self.attention = ST(13)

    def forward(self, obs: torch.Tensor) -> Normal:
        # print(f"obs的形状为{obs.shape}")
        obs = obs.reshape(obs.shape[0], -1, 13)
        x = self.attention(obs)
        mean = self.net(x)
        # print(mean.shape)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [130, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        # self.attention = SelfAttention()
        self.attention = ST(13)

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(f"state的形状为{state.shape}")
        state = state.reshape(state.shape[0], -1, 13)
        x = self.attention(state)
        x = torch.cat((x, action), dim=1)
        # sa = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [128, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)
        # self.attention = SelfAttention()
        self.attention = ST(13)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # print(f"state的形状为{state.shape}")
        state = state.reshape(state.shape[0], -1, 13)
        x = self.attention(state)
        return self.v(x)


class ImplicitQLearning:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_network: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            v_network: nn.Module,
            v_optimizer: torch.optim.Optimizer,
            iql_tau: float = 0.7,
            beta: float = 3.0,
            max_steps: int = 1000000,
            discount: float = 0.99,
            tau: float = 0.005,
            device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
            self,
            next_v: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminals: torch.Tensor,
            log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
            self,
            adv: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


# 可以将TrainConfig里面的参数映射到train方法里面
@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = 78
    action_dim = 2

    file_path = os.path.join(project_root, "dataset", "dataset.hdf5")
    with h5py.File(file_path, 'r') as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    reward_model = TransformerRewardModel(config.env, state_dim, action_dim, structure_type="transformer1",
                                          ensemble_size=3, lr=0.0003, activation="tanh",
                                          d_model=256, nhead=4, num_layers=1, max_seq_len=100,
                                          logger=None, device=config.device)
    reward_model_path = config.reward_model_path
    reward_model.load_model(reward_model_path)
    dataset = reward_from_preference(dataset, reward_model, reward_model_type=config.reward_model_type,
                                     device=config.device)

    print(dataset["rewards"])

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)


    max_action = float(1)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}, Device: {config.device}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    # if config.load_model != "":
    #     policy_file = Path(config.load_model)
    #     trainer.load_state_dict(torch.load(policy_file))
    #     actor = trainer.actor

    # wandb_init(asdict(config))
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        # print(batch)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_actor(
                env,
                actor,
                list1,
                list2,
                list3,
                list4,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
    test_actor(
        env,
        actor,
        device=config.device,
        n_episodes=500,
        seed=config.seed,
    )

if __name__ == "__main__":
    train()
