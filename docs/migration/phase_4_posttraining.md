# المرحلة 4: التدريب اللاحق (Post-Training)
## Phase 4: Post-Training

**المدة:** أسبوعان (الأسبوع 7-8)  
**الهدف:** تقطير المعرفة من 1000 معلم إلى سياسة واحدة

---

## الأسبوع 7: بنية التدريب اللاحق

### اليوم 1-3: تنفيذ Knowledge Distillation

#### المهام:
- [ ] تنفيذ تحميل المعلمين
- [ ] تنفيذ جمع البيانات
- [ ] تنفيذ Behavioral Cloning
- [ ] تنفيذ DAgger

#### التفاصيل:

**4.1 src/raptor/training/post_training.py:**
```python
"""Post-training: Knowledge distillation from teachers to student."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..nn.policy import RaptorPolicy
from ..rl.sac import SAC
from ..rl.replay_buffer import SequenceReplayBuffer
from ..environments.l2f_wrapper import L2FEnv
from ..utils.logging import Logger
from ..utils.checkpoint import CheckpointManager
from ..utils.config import Config

class PostTrainer:
    """Post-training manager for knowledge distillation."""
    
    def __init__(
        self,
        config: Config,
        output_dir: Path,
        teachers_dir: Path,
        dynamics_dir: Path,
        device: str = "cuda"
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.teachers_dir = Path(teachers_dir)
        self.dynamics_dir = Path(dynamics_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load teachers
        self.teachers: List[SAC] = []
        self.dynamics_paths: List[Path] = []
        self._load_teachers()
        
        # Create student
        self.student = RaptorPolicy(
            obs_dim=config.observation.dim,
            hidden_dim=config.student.hidden_dim,
            action_dim=config.action.dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=config.distillation.learning_rate
        )
        
        # Logger
        self.logger = Logger(self.output_dir / "logs", name="post_training")
        
        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(self.output_dir / "checkpoints")
    
    def _load_teachers(self) -> None:
        """Load all teacher policies."""
        teacher_dirs = sorted(self.teachers_dir.glob("teacher_*"))
        dynamics_files = sorted(self.dynamics_dir.glob("*.json"))
        
        num_teachers = min(
            len(teacher_dirs),
            len(dynamics_files),
            self.config.distillation.num_teachers
        )
        
        self.logger.info(f"Loading {num_teachers} teachers...")
        
        for i in tqdm(range(num_teachers), desc="Loading teachers"):
            teacher_dir = teacher_dirs[i]
            dynamics_path = dynamics_files[i]
            
            # Create SAC and load weights
            sac = SAC(
                obs_dim=self.config.observation.dim,
                action_dim=self.config.action.dim,
                device=str(self.device)
            )
            
            final_path = teacher_dir / "final.pt"
            if final_path.exists():
                sac.load(str(final_path))
                self.teachers.append(sac)
                self.dynamics_paths.append(dynamics_path)
        
        self.logger.info(f"Loaded {len(self.teachers)} teachers")
    
    def collect_trajectories(
        self,
        use_student: bool = False,
        num_episodes: int = 10
    ) -> SequenceReplayBuffer:
        """Collect trajectories from teachers (or student for DAgger)."""
        buffer = SequenceReplayBuffer(
            capacity=len(self.teachers) * num_episodes,
            obs_dim=self.config.observation.dim,
            action_dim=self.config.action.dim,
            sequence_length=self.config.distillation.sequence_length
        )
        
        for teacher_idx, (teacher, dynamics_path) in enumerate(
            tqdm(zip(self.teachers, self.dynamics_paths), 
                 total=len(self.teachers),
                 desc="Collecting trajectories")
        ):
            env = L2FEnv(
                drone_config=str(dynamics_path),
                episode_steps=self.config.environment.episode_steps
            )
            
            for _ in range(num_episodes):
                obs, _ = env.reset()
                
                if use_student:
                    self.student.reset()
                
                done = False
                while not done:
                    # Get action from teacher (for labels)
                    teacher_action = teacher.select_action(obs, deterministic=True)
                    
                    # Get action to execute
                    if use_student:
                        with torch.no_grad():
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                            student_action = self.student.evaluate_step(obs_tensor)
                            exec_action = student_action.cpu().numpy().squeeze()
                    else:
                        exec_action = teacher_action
                    
                    # Step environment
                    next_obs, reward, terminated, truncated, _ = env.step(exec_action)
                    done = terminated or truncated
                    
                    # Store with teacher action as label
                    buffer.add(obs, teacher_action, reward, next_obs, done)
                    obs = next_obs
            
            env.close()
        
        return buffer
    
    def train_epoch(
        self,
        buffer: SequenceReplayBuffer,
        epoch: int
    ) -> Dict[str, float]:
        """Train student for one epoch."""
        batch_size = self.config.distillation.batch_size
        num_batches = max(1, len(buffer) // batch_size)
        
        total_loss = 0
        
        for batch_idx in range(num_batches):
            batch = buffer.sample(batch_size, self.device)
            
            # Forward pass through student
            obs = batch['observations']  # (batch, seq, obs_dim)
            teacher_actions = batch['actions']  # (batch, seq, action_dim)
            
            # Process sequence
            batch_size_actual, seq_len, _ = obs.shape
            hidden = self.student.gru.init_hidden(batch_size_actual, self.device)
            
            student_actions = []
            for t in range(seq_len):
                obs_t = obs[:, t, :]  # (batch, obs_dim)
                action_t, hidden = self.student(obs_t, hidden)
                student_actions.append(action_t)
            
            student_actions = torch.stack(student_actions, dim=1)  # (batch, seq, action_dim)
            
            # Compute loss (MSE between student and teacher actions)
            loss = F.mse_loss(student_actions, teacher_actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def evaluate(self) -> float:
        """Evaluate student policy."""
        total_reward = 0
        num_eval = min(100, len(self.teachers))
        
        for i in range(num_eval):
            dynamics_path = self.dynamics_paths[i]
            env = L2FEnv(
                drone_config=str(dynamics_path),
                episode_steps=self.config.environment.episode_steps
            )
            
            obs, _ = env.reset()
            self.student.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action = self.student.evaluate_step(obs_tensor)
                    action = action.cpu().numpy().squeeze()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_reward += episode_reward
            env.close()
        
        return total_reward / num_eval
    
    def train(self) -> None:
        """Run full post-training."""
        n_epochs = self.config.distillation.n_epochs
        teacher_forcing_epochs = self.config.distillation.epoch_teacher_forcing
        num_episodes = self.config.distillation.get('num_episodes', 10)
        
        best_eval_reward = -float('inf')
        
        for epoch in tqdm(range(n_epochs), desc="Post-training"):
            # Collect trajectories
            use_student = epoch >= teacher_forcing_epochs
            buffer = self.collect_trajectories(
                use_student=use_student,
                num_episodes=num_episodes
            )
            
            # Train epoch
            metrics = self.train_epoch(buffer, epoch)
            
            # Log
            self.logger.log_scalar("train/loss", metrics['loss'], epoch)
            self.logger.set_step(epoch)
            
            # Evaluate
            if epoch % 10 == 0:
                eval_reward = self.evaluate()
                self.logger.log_scalar("eval/reward", eval_reward, epoch)
                self.logger.info(f"Epoch {epoch}: loss={metrics['loss']:.4f}, eval_reward={eval_reward:.2f}")
                
                # Save best
                is_best = eval_reward > best_eval_reward
                if is_best:
                    best_eval_reward = eval_reward
                
                self.checkpoint_mgr.save(
                    self.student,
                    self.optimizer,
                    epoch=epoch,
                    metrics={'eval_reward': eval_reward},
                    is_best=is_best
                )
        
        # Final save
        torch.save(self.student.state_dict(), self.output_dir / "final_policy.pt")
        self.logger.info(f"Training complete. Best eval reward: {best_eval_reward:.2f}")
        self.logger.close()
```

**4.2 scripts/train_student.py:**
```python
#!/usr/bin/env python
"""Script to train student policy (post-training)."""
import argparse
from pathlib import Path
from raptor.training.post_training import PostTrainer
from raptor.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train student policy")
    parser.add_argument("--config", type=str, default="configs/post_training.yaml")
    parser.add_argument("--output-dir", type=str, default="experiments/post_training")
    parser.add_argument("--teachers-dir", type=str, default="experiments/pre_training/checkpoints")
    parser.add_argument("--dynamics-dir", type=str, default="data/dynamics")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    
    # Create trainer
    trainer = PostTrainer(
        config=config,
        output_dir=Path(args.output_dir),
        teachers_dir=Path(args.teachers_dir),
        dynamics_dir=Path(args.dynamics_dir),
        device=args.device
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

#### المخرجات:
- [x] src/raptor/training/post_training.py
- [x] scripts/train_student.py

---

## الأسبوع 8: تشغيل التدريب اللاحق

### اليوم 1-5: تشغيل ومراقبة التدريب

#### المهام:
- [ ] تشغيل التدريب اللاحق
- [ ] مراقبة TensorBoard
- [ ] التحقق من جودة السياسة النهائية
- [ ] حل أي مشاكل

#### الأوامر:
```bash
# تدريب السياسة الطالب
python scripts/train_student.py --device cuda

# مراقبة TensorBoard
tensorboard --logdir experiments/post_training/logs
```

#### معايير النجاح للمرحلة 4:
- [ ] سياسة طالب مدربة
- [ ] متوسط المكافأة > 250
- [ ] السياسة تعمل على جميع تكوينات الديناميكا
- [ ] Checkpoint نهائي محفوظ

---

