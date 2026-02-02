# المرحلة 2: البنية التحتية للتعلم المعزز
## Phase 2: RL Infrastructure

**المدة:** أسبوعان (الأسبوع 3-4)  
**الهدف:** تنفيذ خوارزمية SAC وتكامل بيئة L2F

---

## الأسبوع 3: تكامل بيئة L2F

### اليوم 1-2: Gymnasium Wrapper

#### المهام:
- [ ] تنفيذ L2F Gymnasium Wrapper
- [ ] تنفيذ Observation Normalization
- [ ] تنفيذ Action Scaling

#### التفاصيل:

**2.1 src/raptor/environments/l2f_wrapper.py:**
```python
"""L2F Gymnasium Wrapper."""
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import l2f
from l2f import vector8 as vector

class L2FEnv(gym.Env):
    """
    Gymnasium wrapper for L2F environment.
    
    Observation Space: Box(22,) - position, rotation matrix, velocities, prev action
    Action Space: Box(4,) - normalized motor commands [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        drone_config: Optional[str] = None,
        dt: float = 0.01,
        episode_steps: int = 500,
        render_mode: Optional[str] = None,
        seed: int = 0
    ):
        super().__init__()
        
        self.dt = dt
        self.episode_steps = episode_steps
        self.render_mode = render_mode
        
        # Initialize L2F
        self.device = l2f.Device()
        self.rng = vector.VectorRng()
        self.env = vector.VectorEnvironment()
        self.params = vector.VectorParameters()
        self.state = vector.VectorState()
        self.next_state = vector.VectorState()
        
        vector.initialize_rng(self.device, self.rng, seed)
        vector.initialize_environment(self.device, self.env)
        
        # Load drone config if provided
        if drone_config is not None:
            self._load_drone_config(drone_config)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Internal state
        self._step_count = 0
        self._observation = np.zeros(22, dtype=np.float32)
    
    def _load_drone_config(self, config_path: str) -> None:
        """Load drone configuration from JSON."""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        params_json = json.dumps(config)
        self.params = l2f.parameters_from_json(
            self.device, self.env, params_json
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            vector.initialize_rng(self.device, self.rng, seed)
        
        vector.sample_initial_parameters(
            self.device, self.env, self.params, self.rng
        )
        vector.sample_initial_state(
            self.device, self.env, self.params, self.state, self.rng
        )
        
        self._step_count = 0
        self._get_observation()
        
        return self._observation.copy(), {}
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take environment step."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Step environment
        dts = vector.step(
            self.device, self.env, self.params,
            self.state, action.reshape(1, 4),
            self.next_state, self.rng
        )
        
        self.state.assign(self.next_state)
        self._step_count += 1
        
        # Get observation and reward
        self._get_observation()
        reward = self._compute_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self.episode_steps
        
        info = {
            'step': self._step_count,
            'dt': dts[0]
        }
        
        return self._observation.copy(), reward, terminated, truncated, info
    
    def _get_observation(self) -> None:
        """Get observation from state."""
        obs_buffer = np.zeros((1, self.env.OBSERVATION_DIM), dtype=np.float32)
        vector.observe(
            self.device, self.env, self.params,
            self.state, obs_buffer, self.rng
        )
        self._observation = obs_buffer[0, :22]
    
    def _compute_reward(self) -> float:
        """Compute reward."""
        # Position error
        pos = self._observation[:3]
        pos_error = np.sum(pos ** 2)
        
        # Orientation error (deviation from identity)
        rot = self._observation[3:12].reshape(3, 3)
        rot_error = np.sum((rot - np.eye(3)) ** 2)
        
        # Velocity errors
        lin_vel = self._observation[12:15]
        ang_vel = self._observation[15:18]
        vel_error = np.sum(lin_vel ** 2) + np.sum(ang_vel ** 2)
        
        # Action magnitude
        action = self._observation[18:22]
        action_error = np.sum(action ** 2)
        
        # Reward
        reward = 0.1 * (1.0 - (
            10.0 * pos_error +
            2.5 * rot_error +
            0.05 * vel_error +
            0.1 * action_error
        ))
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        pos = self._observation[:3]
        if np.linalg.norm(pos) > 2.0:
            return True
        return False
    
    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode == "rgb_array":
            # Return placeholder for now
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None
    
    def close(self) -> None:
        """Close environment."""
        pass


def make_vec_env(
    n_envs: int = 8,
    drone_config: Optional[str] = None,
    seed: int = 0
) -> gym.vector.VectorEnv:
    """Create vectorized environment."""
    def make_env(env_seed: int):
        def _init():
            return L2FEnv(
                drone_config=drone_config,
                seed=env_seed
            )
        return _init
    
    envs = [make_env(seed + i) for i in range(n_envs)]
    return gym.vector.SyncVectorEnv(envs)
```

#### المخرجات:
- [x] src/raptor/environments/l2f_wrapper.py

---

### اليوم 3-4: Replay Buffer

#### المهام:
- [ ] تنفيذ Replay Buffer أساسي
- [ ] تنفيذ Sequence Replay Buffer للـ GRU
- [ ] اختبارات

#### التفاصيل:

**2.2 src/raptor/rl/replay_buffer.py:**
```python
"""Replay Buffer implementations."""
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Transition:
    """Single transition."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool

class ReplayBuffer:
    """Standard replay buffer."""
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """Add transition to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'observations': torch.FloatTensor(
                self.observations[indices]
            ).to(device),
            'actions': torch.FloatTensor(
                self.actions[indices]
            ).to(device),
            'rewards': torch.FloatTensor(
                self.rewards[indices]
            ).to(device),
            'next_observations': torch.FloatTensor(
                self.next_observations[indices]
            ).to(device),
            'dones': torch.FloatTensor(
                self.dones[indices]
            ).to(device),
        }
    
    def __len__(self) -> int:
        return self.size


class SequenceReplayBuffer:
    """Replay buffer for sequence-based training (GRU)."""
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        sequence_length: int
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        
        # Store episodes
        self.episodes: list = []
        self.current_episode: list = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """Add transition to current episode."""
        self.current_episode.append(Transition(
            observation=obs.copy(),
            action=action.copy(),
            reward=reward,
            next_observation=next_obs.copy(),
            done=done
        ))
        
        if done:
            self._finish_episode()
    
    def _finish_episode(self) -> None:
        """Finish current episode and add to buffer."""
        if len(self.current_episode) >= self.sequence_length:
            self.episodes.append(self.current_episode)
            
            # Remove old episodes if over capacity
            while len(self.episodes) > self.capacity:
                self.episodes.pop(0)
        
        self.current_episode = []
    
    def sample(
        self,
        batch_size: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Sample batch of sequences."""
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_dones = []
        
        for _ in range(batch_size):
            # Sample random episode
            episode_idx = np.random.randint(0, len(self.episodes))
            episode = self.episodes[episode_idx]
            
            # Sample random starting point
            max_start = len(episode) - self.sequence_length
            start_idx = np.random.randint(0, max_start + 1)
            
            # Extract sequence
            seq_obs = np.array([
                t.observation for t in episode[start_idx:start_idx + self.sequence_length]
            ])
            seq_actions = np.array([
                t.action for t in episode[start_idx:start_idx + self.sequence_length]
            ])
            seq_rewards = np.array([
                t.reward for t in episode[start_idx:start_idx + self.sequence_length]
            ])
            seq_next_obs = np.array([
                t.next_observation for t in episode[start_idx:start_idx + self.sequence_length]
            ])
            seq_dones = np.array([
                float(t.done) for t in episode[start_idx:start_idx + self.sequence_length]
            ])
            
            batch_obs.append(seq_obs)
            batch_actions.append(seq_actions)
            batch_rewards.append(seq_rewards)
            batch_next_obs.append(seq_next_obs)
            batch_dones.append(seq_dones)
        
        return {
            'observations': torch.FloatTensor(
                np.array(batch_obs)
            ).to(device),  # (batch, seq, obs_dim)
            'actions': torch.FloatTensor(
                np.array(batch_actions)
            ).to(device),  # (batch, seq, action_dim)
            'rewards': torch.FloatTensor(
                np.array(batch_rewards)
            ).to(device),  # (batch, seq)
            'next_observations': torch.FloatTensor(
                np.array(batch_next_obs)
            ).to(device),  # (batch, seq, obs_dim)
            'dones': torch.FloatTensor(
                np.array(batch_dones)
            ).to(device),  # (batch, seq)
        }
    
    def __len__(self) -> int:
        return len(self.episodes)
```

#### المخرجات:
- [x] src/raptor/rl/replay_buffer.py

---

### اليوم 5: اختبارات البيئة

#### المهام:
- [ ] اختبار L2F Wrapper
- [ ] اختبار Replay Buffer
- [ ] تكامل مع Gymnasium

#### التفاصيل:

**2.3 tests/test_environment.py:**
```python
"""Tests for environment."""
import pytest
import numpy as np
from raptor.environments.l2f_wrapper import L2FEnv, make_vec_env

class TestL2FEnv:
    def test_observation_space(self):
        env = L2FEnv()
        assert env.observation_space.shape == (22,)
    
    def test_action_space(self):
        env = L2FEnv()
        assert env.action_space.shape == (4,)
    
    def test_reset(self):
        env = L2FEnv()
        obs, info = env.reset()
        assert obs.shape == (22,)
        assert isinstance(info, dict)
    
    def test_step(self):
        env = L2FEnv()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (22,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_episode_length(self):
        env = L2FEnv(episode_steps=100)
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                assert i == 99
                break

class TestVecEnv:
    def test_make_vec_env(self):
        vec_env = make_vec_env(n_envs=4)
        assert vec_env.num_envs == 4
    
    def test_vec_reset(self):
        vec_env = make_vec_env(n_envs=4)
        obs, info = vec_env.reset()
        assert obs.shape == (4, 22)
    
    def test_vec_step(self):
        vec_env = make_vec_env(n_envs=4)
        vec_env.reset()
        actions = np.random.uniform(-1, 1, (4, 4))
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        assert obs.shape == (4, 22)
        assert rewards.shape == (4,)
```

#### المخرجات:
- [x] tests/test_environment.py

---

## الأسبوع 4: خوارزمية SAC

### اليوم 1-3: تنفيذ SAC

#### المهام:
- [ ] تنفيذ Actor Network
- [ ] تنفيذ Critic Network
- [ ] تنفيذ SAC Algorithm

#### التفاصيل:

**2.4 src/raptor/rl/networks.py:**
```python
"""Neural networks for SAC."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    """SAC Actor (Policy) Network."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def sample(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log probability with correction for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Get action for inference."""
        mean, log_std = self.forward(obs)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        return torch.tanh(x_t)


class Critic(nn.Module):
    """SAC Critic (Q-function) Network."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Q1
        self.q1_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.q2_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Q1 and Q2."""
        x = torch.cat([obs, action], dim=-1)
        
        # Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        
        # Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2
    
    def q1(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning only Q1."""
        x = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)
```

**2.5 src/raptor/rl/sac.py:**
```python
"""Soft Actor-Critic (SAC) Algorithm."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer

class SAC:
    """Soft Actor-Critic algorithm."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy: Optional[float] = None,
        lr: float = 3e-4,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Target entropy
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        
        # Networks
        self.actor = Actor(obs_dim, action_dim, actor_hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim, critic_hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, critic_hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Learnable temperature
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            action = self.actor.get_action(obs_tensor, deterministic)
            return action.cpu().numpy().squeeze()
    
    def update(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Update networks."""
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards'].unsqueeze(-1)
        next_obs = batch['next_observations']
        dones = batch['dones'].unsqueeze(-1)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target_q = rewards + self.gamma * (1 - dones) * (q_target - self.alpha * next_log_probs)
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # Update target networks
        self._soft_update()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'q_value': q1.mean().item()
        }
    
    def _soft_update(self) -> None:
        """Soft update target networks."""
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str) -> None:
        """Save model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.alpha = self.log_alpha.exp().item()
```

#### المخرجات:
- [x] src/raptor/rl/networks.py
- [x] src/raptor/rl/sac.py

---

### اليوم 4-5: اختبارات SAC

#### المهام:
- [ ] اختبار Actor/Critic
- [ ] اختبار SAC update
- [ ] تجربة تدريب بسيطة

#### التفاصيل:

**2.6 tests/test_sac.py:**
```python
"""Tests for SAC algorithm."""
import pytest
import torch
import numpy as np
from raptor.rl.sac import SAC
from raptor.rl.replay_buffer import ReplayBuffer

class TestSAC:
    def test_select_action(self):
        sac = SAC(obs_dim=22, action_dim=4, device="cpu")
        obs = np.random.randn(22).astype(np.float32)
        action = sac.select_action(obs)
        assert action.shape == (4,)
        assert np.all(action >= -1) and np.all(action <= 1)
    
    def test_update(self):
        sac = SAC(obs_dim=22, action_dim=4, device="cpu")
        
        # Create fake batch
        batch = {
            'observations': torch.randn(32, 22),
            'actions': torch.randn(32, 4),
            'rewards': torch.randn(32),
            'next_observations': torch.randn(32, 22),
            'dones': torch.zeros(32),
        }
        
        metrics = sac.update(batch)
        assert 'critic_loss' in metrics
        assert 'actor_loss' in metrics
        assert 'alpha' in metrics
    
    def test_save_load(self, tmp_path):
        sac = SAC(obs_dim=22, action_dim=4, device="cpu")
        path = tmp_path / "sac.pt"
        sac.save(str(path))
        
        sac2 = SAC(obs_dim=22, action_dim=4, device="cpu")
        sac2.load(str(path))
        
        # Check weights are same
        for p1, p2 in zip(sac.actor.parameters(), sac2.actor.parameters()):
            assert torch.allclose(p1, p2)
```

#### معايير النجاح للمرحلة 2:
- [ ] L2F Wrapper يعمل مع Gymnasium
- [ ] Replay Buffer يعمل بشكل صحيح
- [ ] SAC يتدرب بدون أخطاء
- [ ] جميع الاختبارات تمر

---

