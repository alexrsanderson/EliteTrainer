import numpy as np
import asyncio
import gymnasium as gym
from gymnasium.spaces import Space, Box
from gymnasium.utils.env_checker import check_env
from poke_env.player import (
    MaxBasePowerPlayer,
    EnvPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    Gen9EnvSinglePlayer,
    ObsType
)
from poke_env.environment import (
    Battle,
    AbstractBattle,
)
from poke_env import (
    AccountConfiguration,
    ShowdownServerConfiguration,
    LocalhostServerConfiguration
)
from typing import Dict, List, Optional, Union
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.data import GenData
from threading import Lock
from gymnasium.core import ObsType, Env
from gymnasium.envs.registration import register
import torch
from torch import nn
from collections import deque
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tabulate import tabulate
class EliteTrainer(EnvPlayer, GenData):
    _ACTION_SPACE = list(range(5*4+6))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = 0.9
        self.batch_size = 32
        self.device = 'cpu'
        self.net = PokeNet(self.observation_space.shape, self.action_space.n)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = 0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.exploration_rate = 1
        self.decay = 0.01
        self.explore_min = 0.01
        self.curr_step = 0
        self.burnin = 1
        self.learn_every = 1
        self.sync_every = 1
        self.action_id = 0
    def calc_reward(self, last_battle: AbstractBattle, current_battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=6, hp_value=1, victory_value=30, status_value=0.5
        )
    def optimal_move(self, state) -> int:
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(26)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)            
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, dim=1).item()
        self.exploration_rate *= self.decay
        self.exploration_rate = max(self.explore_min, self.exploration_rate)
        self.curr_step += 1
        self.action_id = action_idx
    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        return Gen9EnvSinglePlayer.action_to_move(self, self.action_id, battle)
    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    type_1=battle.opponent_active_pokemon.type_1,
                    type_2=battle.opponent_active_pokemon.type_2,
                    type_chart=self.load_type_chart(9)
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        opponent_moves_power = -np.ones(4)
        opp_moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.opponent_active_pokemon.moves.values()):
            opponent_moves_power[i] = (move.base_power / 100)
            if move.type:
                opp_moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    type_1=battle.active_pokemon.type_1,
                    type_2=battle.active_pokemon.type_2,
                    type_chart=self.load_type_chart(9)
                )
        # Final vector with 18 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                opponent_moves_power, 
                opp_moves_dmg_multiplier,
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 3, 3, 3, 3, 4, 4, 4 ,4]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    def cache(self, state, next_state, action, reward, terminated):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated)
        self.memory.add(TensorDict({'state': state, 'next_state':next_state, 
        'action':action, 'reward':reward, 'terminated':terminated},batch_size=[]))
    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, terminated = (batch.get(key) for key in ('state', 'next_state', 'action',
        'reward', 'terminated'))
        return state, next_state, action, reward, terminated
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every !=0:
            return None, None
        state, next_state, action, reward, terminated = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, terminated)
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(),loss)
    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online').view(self.batch_size, -1)
        return current_Q
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model = 'online')
        optimal_move = torch.argmax(next_state_Q, dim=1, keepdim=True)
        next_Q_values = self.net(next_state, model='target')
        return (reward.view(self.batch_size, -1) + (1 - done.float()).view(self.batch_size, -1) * self.gamma * next_Q_values)
    def update_Q_online(self, td_estimate, td_target):
        #print(td_estimate, td_estimate.shape, td_target, td_target.shape)
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
class PokeNet(nn.Module):
    def __init__(self, input, output):
        super(PokeNet, self).__init__()
        output_size = int(output)
        input_size = int(np.prod(input))
        self.online = self._build_dqn(input_size, output_size)
        self.target = self._build_dqn(input_size, output_size)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
    def forward(self, input, model):
        if model == 'online': 
            return self.online(input)
        elif model == 'target':
            return self.target(input)
    def _build_dqn(self, input, output_size):
        return nn.Sequential(
            nn.Linear(input,64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
async def main():
    train_env = EliteTrainer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True, 
        start_timer_on_battle_start = True,
        opponent = MaxBasePowerPlayer(battle_format='gen9randombattle')
    )
    def run(env, episodes):
        for ep in range(episodes):
            state = env.reset()
            while True:
                #print(type(state))
                env.optimal_move(state)
                #print(f"Optimal move (integer): {optimal_move} (type: {type(optimal_move)})")
                action = env.action_to_move(env.action_id, env.current_battle)  
                next_state, reward, terminated, truncated, info = env.step(action)
                #print(f"Action to move: {action} (type: {type(action)})")
                env.cache(state, next_state, env.action_id, reward, terminated)
                q, loss = env.learn()
                state = next_state
                env.render()
                if terminated or truncated:
                    break
        print(
        f"Double DQN Evaluation: {env.n_won_battles} victories out of {env.n_finished_battles} episodes")
        train_env.reset_env(restart=True)
        train_env.close(purge=True)
    n_episodes = 1000
    run(train_env, n_episodes)
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())