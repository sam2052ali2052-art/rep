# المرحلة 3: التدريب المسبق (Pre-Training)
## Phase 3: Pre-Training

**المدة:** أسبوعان (الأسبوع 5-6)  
**الهدف:** تدريب 1000 سياسة معلم

---

## الأسبوع 5: بنية التدريب المسبق

### اليوم 1-2: توليد معلمات الديناميكا

#### المهام:
- [ ] تنفيذ Domain Randomization Sampler
- [ ] توليد 1000 ملف JSON
- [ ] التحقق من التوزيع

#### التفاصيل:

**3.1 src/raptor/training/dynamics_sampler.py:**
```python
"""Dynamics parameter sampling for domain randomization."""
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DomainRandomizationConfig:
    """Domain randomization configuration."""
    thrust_to_weight: tuple = (1.5, 5.0)
    torque_to_inertia: tuple = (40, 1200)
    mass: tuple = (0.02, 5.0)
    rotor_time_constant_rising: tuple = (0.03, 0.30)
    rotor_time_constant_falling: tuple = (0.07, 0.30)
    rotor_torque_constant: tuple = (0.005, 0.05)
    disturbance_force: tuple = (0, 0.3)

def sample_dynamics_parameters(
    config: DomainRandomizationConfig,
    rng: np.random.Generator
) -> Dict[str, Any]:
    """Sample random dynamics parameters."""
    
    # Sample basic parameters
    mass = rng.uniform(*config.mass)
    thrust_to_weight = rng.uniform(*config.thrust_to_weight)
    torque_to_inertia = rng.uniform(*config.torque_to_inertia)
    
    # Compute derived parameters
    gravity = 9.81
    total_thrust = mass * gravity * thrust_to_weight
    thrust_per_motor = total_thrust / 4
    
    # Rotor parameters
    rotor_time_rising = rng.uniform(*config.rotor_time_constant_rising)
    rotor_time_falling = rng.uniform(*config.rotor_time_constant_falling)
    rotor_torque = rng.uniform(*config.rotor_torque_constant)
    
    # Inertia (simplified model based on mass)
    arm_length = 0.1 * (mass / 0.5) ** (1/3)  # Scale with mass
    inertia = mass * arm_length ** 2 / 12
    
    # Build parameters dict
    params = {
        "dynamics": {
            "rotor_positions": [
                [arm_length, -arm_length, 0.0],
                [-arm_length, -arm_length, 0.0],
                [-arm_length, arm_length, 0.0],
                [arm_length, arm_length, 0.0]
            ],
            "rotor_thrust_directions": [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0]
            ],
            "rotor_torque_directions": [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0]
            ],
            "mass": mass,
            "gravity": [0.0, 0.0, -gravity],
            "J": [
                [inertia, 0.0, 0.0],
                [0.0, inertia, 0.0],
                [0.0, 0.0, inertia * 2]
            ],
            "rotor_torque_constants": [rotor_torque] * 4,
            "rotor_time_constants_rising": [rotor_time_rising] * 4,
            "rotor_time_constants_falling": [rotor_time_falling] * 4
        },
        "mdp": {
            "init": {
                "max_position": 0.5,
                "max_angle": 1.57,
                "max_linear_velocity": 1.0,
                "max_angular_velocity": 1.0
            },
            "reward": {
                "scale": 0.1,
                "constant": 1.0,
                "termination_penalty": -100,
                "position": 10.0,
                "orientation": 2.5,
                "linear_velocity": 0.05,
                "action": 0.1,
                "d_action": 1.0
            }
        },
        "metadata": {
            "thrust_to_weight": thrust_to_weight,
            "torque_to_inertia": torque_to_inertia,
            "mass": mass
        }
    }
    
    return params

def generate_dynamics_files(
    output_dir: Path,
    num_files: int = 1000,
    seed: int = 42
) -> None:
    """Generate dynamics parameter files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = DomainRandomizationConfig()
    rng = np.random.default_rng(seed)
    
    for i in range(num_files):
        params = sample_dynamics_parameters(config, rng)
        
        filepath = output_dir / f"{i:04d}.json"
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    print(f"Generated {num_files} dynamics parameter files in {output_dir}")
```

#### المخرجات:
- [x] src/raptor/training/dynamics_sampler.py

---

### اليوم 3-5: حلقة التدريب المسبق

#### المهام:
- [ ] تنفيذ حلقة التدريب
- [ ] تنفيذ التقييم
- [ ] تنفيذ حفظ Checkpoints

#### التفاصيل:

**3.2 src/raptor/training/pre_training.py:**
```python
"""Pre-training: Train teacher policies."""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

from ..rl.sac import SAC
from ..rl.replay_buffer import ReplayBuffer
from ..environments.l2f_wrapper import L2FEnv
from ..utils.logging import Logger
from ..utils.checkpoint import CheckpointManager
from ..utils.config import Config, load_config

class PreTrainer:
    """Pre-training manager for teacher policies."""
    
    def __init__(
        self,
        config: Config,
        output_dir: Path,
        device: str = "cuda"
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
    
    def train_teacher(
        self,
        teacher_id: int,
        dynamics_path: Path
    ) -> Dict[str, Any]:
        """Train a single teacher policy."""
        
        # Setup logging
        logger = Logger(
            self.output_dir / "logs" / f"teacher_{teacher_id:04d}",
            name=f"teacher_{teacher_id}"
        )
        
        # Setup checkpoint manager
        checkpoint_mgr = CheckpointManager(
            self.output_dir / "checkpoints" / f"teacher_{teacher_id:04d}"
        )
        
        # Create environment
        env = L2FEnv(
            drone_config=str(dynamics_path),
            dt=self.config.environment.dt,
            episode_steps=self.config.environment.episode_steps
        )
        
        # Create SAC agent
        sac = SAC(
            obs_dim=self.config.observation.dim,
            action_dim=self.config.action.dim,
            actor_hidden_dim=self.config.sac.actor_hidden_dim,
            critic_hidden_dim=self.config.sac.critic_hidden_dim,
            gamma=self.config.sac.gamma,
            tau=self.config.sac.tau,
            target_entropy=self.config.sac.target_entropy,
            lr=self.config.sac.learning_rate,
            device=self.device
        )
        
        # Create replay buffer
        buffer = ReplayBuffer(
            capacity=1000000,
            obs_dim=self.config.observation.dim,
            action_dim=self.config.action.dim
        )
        
        # Training loop
        total_steps = self.config.training.steps_per_teacher
        warmup_steps = self.config.sac.n_warmup_steps
        eval_freq = self.config.training.eval_frequency
        checkpoint_freq = self.config.training.checkpoint_frequency
        batch_size = self.config.sac.actor_batch_size
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        best_eval_reward = -float('inf')
        
        for step in tqdm(range(total_steps), desc=f"Teacher {teacher_id}"):
            # Select action
            if step < warmup_steps:
                action = env.action_space.sample()
            else:
                action = sac.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.add(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update
            if step >= warmup_steps and len(buffer) >= batch_size:
                batch = buffer.sample(batch_size, torch.device(self.device))
                metrics = sac.update(batch)
                
                # Log metrics
                if step % 1000 == 0:
                    logger.log_scalar("train/critic_loss", metrics['critic_loss'], step)
                    logger.log_scalar("train/actor_loss", metrics['actor_loss'], step)
                    logger.log_scalar("train/alpha", metrics['alpha'], step)
                    logger.log_scalar("train/q_value", metrics['q_value'], step)
            
            # Episode end
            if done:
                logger.log_scalar("train/episode_reward", episode_reward, step)
                logger.log_scalar("train/episode_length", episode_length, step)
                logger.info(f"Episode {episode_count}: reward={episode_reward:.2f}, length={episode_length}")
                
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
            else:
                obs = next_obs
            
            # Evaluation
            if step > 0 and step % eval_freq == 0:
                eval_reward = self._evaluate(env, sac)
                logger.log_scalar("eval/reward", eval_reward, step)
                logger.info(f"Step {step}: eval_reward={eval_reward:.2f}")
                
                is_best = eval_reward > best_eval_reward
                if is_best:
                    best_eval_reward = eval_reward
            
            # Checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_mgr.save(
                    sac.actor,
                    epoch=0,
                    step=step,
                    metrics={'eval_reward': best_eval_reward},
                    is_best=False
                )
        
        # Final save
        final_path = self.output_dir / "checkpoints" / f"teacher_{teacher_id:04d}" / "final.pt"
        sac.save(str(final_path))
        
        logger.close()
        env.close()
        
        return {
            'teacher_id': teacher_id,
            'best_eval_reward': best_eval_reward,
            'final_path': str(final_path)
        }
    
    def _evaluate(
        self,
        env: L2FEnv,
        sac: SAC,
        num_episodes: int = 10
    ) -> float:
        """Evaluate policy."""
        total_reward = 0
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = sac.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def train_all_teachers(
        self,
        dynamics_dir: Path,
        num_teachers: Optional[int] = None
    ) -> None:
        """Train all teacher policies."""
        dynamics_dir = Path(dynamics_dir)
        dynamics_files = sorted(dynamics_dir.glob("*.json"))
        
        if num_teachers is not None:
            dynamics_files = dynamics_files[:num_teachers]
        
        results = []
        for i, dynamics_path in enumerate(dynamics_files):
            result = self.train_teacher(i, dynamics_path)
            results.append(result)
            
            # Save results
            import json
            with open(self.output_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)
```

**3.3 scripts/train_teachers.py:**
```python
#!/usr/bin/env python
"""Script to train teacher policies."""
import argparse
from pathlib import Path
from raptor.training.pre_training import PreTrainer
from raptor.training.dynamics_sampler import generate_dynamics_files
from raptor.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train teacher policies")
    parser.add_argument("--config", type=str, default="configs/pre_training.yaml")
    parser.add_argument("--output-dir", type=str, default="experiments/pre_training")
    parser.add_argument("--dynamics-dir", type=str, default="data/dynamics")
    parser.add_argument("--num-teachers", type=int, default=None)
    parser.add_argument("--generate-dynamics", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Generate dynamics files if needed
    if args.generate_dynamics:
        generate_dynamics_files(Path(args.dynamics_dir), num_files=1000)
    
    # Load config
    config = load_config(Path(args.config))
    
    # Create trainer
    trainer = PreTrainer(
        config=config,
        output_dir=Path(args.output_dir),
        device=args.device
    )
    
    # Train teachers
    trainer.train_all_teachers(
        dynamics_dir=Path(args.dynamics_dir),
        num_teachers=args.num_teachers
    )

if __name__ == "__main__":
    main()
```

#### المخرجات:
- [x] src/raptor/training/pre_training.py
- [x] scripts/train_teachers.py

---

## الأسبوع 6: تشغيل التدريب المسبق

### اليوم 1-5: تشغيل ومراقبة التدريب

#### المهام:
- [ ] تشغيل التدريب على GPU
- [ ] مراقبة TensorBoard
- [ ] التحقق من جودة المعلمين
- [ ] حل أي مشاكل

#### الأوامر:
```bash
# توليد معلمات الديناميكا
python scripts/train_teachers.py --generate-dynamics

# تدريب 10 معلمين للاختبار
python scripts/train_teachers.py --num-teachers 10 --device cuda

# تدريب جميع المعلمين
python scripts/train_teachers.py --device cuda

# مراقبة TensorBoard
tensorboard --logdir experiments/pre_training/logs
```

#### معايير النجاح للمرحلة 3:
- [ ] 1000 معلم مدرب
- [ ] متوسط المكافأة > 200 لكل معلم
- [ ] Checkpoints محفوظة
- [ ] لا أخطاء في التدريب

---

