# المرحلة 1: الأساس وهيكل المشروع
## Phase 1: Foundation & Project Structure

**المدة:** أسبوعان (الأسبوع 1-2)  
**الهدف:** إنشاء هيكل المشروع وتنفيذ الشبكة العصبية الأساسية

---

## الأسبوع 1: إعداد المشروع والشبكة العصبية

### اليوم 1-2: إنشاء هيكل المشروع

#### المهام:
- [ ] إنشاء مستودع Git جديد
- [ ] إعداد هيكل المجلدات
- [ ] إنشاء `pyproject.toml`
- [ ] إعداد بيئة التطوير الافتراضية

#### التفاصيل:

**1.1 إنشاء المستودع:**
```bash
mkdir raptor-python
cd raptor-python
git init
git checkout -b main
```

**1.2 هيكل المجلدات:**
```
raptor-python/
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── pre_training.yaml
│   ├── post_training.yaml
│   └── drones/
│       ├── crazyflie.json
│       ├── x500.json
│       └── custom.json
├── src/
│   └── raptor/
│       ├── __init__.py
│       ├── nn/
│       ├── rl/
│       ├── environments/
│       ├── training/
│       ├── export/
│       └── utils/
├── tests/
├── scripts/
└── checkpoints/
```

**1.3 pyproject.toml:**
```toml
[project]
name = "raptor-python"
version = "0.1.0"
description = "Raptor Foundation Policy for Quadrotor Control"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "l2f==2.0.18",
    "stable-baselines3>=2.0.0",
    "gymnasium>=0.29.0",
    "tensorboard>=2.14.0",
    "onnx>=1.14.0",
    "h5py>=3.9.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
export = [
    "onnx-tf>=1.10.0",
    "tensorflow>=2.14.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

#### المخرجات:
- [x] مستودع Git مُهيأ
- [x] هيكل مجلدات كامل
- [x] ملف pyproject.toml
- [x] بيئة افتراضية تعمل

---

### اليوم 3-4: تنفيذ طبقات الشبكة العصبية

#### المهام:
- [ ] تنفيذ طبقة Dense
- [ ] تنفيذ طبقة GRU
- [ ] تنفيذ نموذج Sequential
- [ ] تنفيذ RaptorPolicy

#### التفاصيل:

**1.4 src/raptor/nn/dense.py:**
```python
"""Dense (Linear) layer implementation."""
import torch
import torch.nn as nn
from typing import Optional

class Dense(nn.Module):
    """Dense layer with optional activation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = None,
        bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "identity" or activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))
```

**1.5 src/raptor/nn/gru.py:**
```python
"""GRU layer implementation compatible with Raptor."""
import torch
import torch.nn as nn
from typing import Tuple, Optional

class GRULayer(nn.Module):
    """GRU layer with hidden state management."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Hidden state of shape (1, batch, hidden_size)
        
        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size)
            hidden: New hidden state
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        
        output, hidden = self.gru(x, hidden)
        return output, hidden
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
```

**1.6 src/raptor/nn/policy.py:**
```python
"""Raptor Policy Network."""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .dense import Dense
from .gru import GRULayer

class RaptorPolicy(nn.Module):
    """
    Raptor Foundation Policy Network.
    
    Architecture: Dense(22→16, ReLU) → GRU(16) → Dense(16→4, Identity)
    """
    
    def __init__(
        self,
        obs_dim: int = 22,
        hidden_dim: int = 16,
        action_dim: int = 4
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Network layers
        self.input_layer = Dense(obs_dim, hidden_dim, activation="relu")
        self.gru = GRULayer(hidden_dim, hidden_dim)
        self.output_layer = Dense(hidden_dim, action_dim, activation="identity")
        
        # Hidden state
        self._hidden: Optional[torch.Tensor] = None
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            hidden: Hidden state of shape (1, batch, hidden_dim)
        
        Returns:
            action: Action tensor of shape (batch, action_dim)
            hidden: New hidden state
        """
        # Input layer
        x = self.input_layer(obs)
        
        # Add sequence dimension for GRU
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # GRU layer
        x, hidden = self.gru(x, hidden)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch, hidden_dim)
        
        # Output layer
        action = self.output_layer(x)
        
        return action, hidden
    
    def reset(self, batch_size: int = 1) -> None:
        """Reset hidden state."""
        self._hidden = None
    
    def evaluate_step(
        self,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate single step (for inference).
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
        
        Returns:
            action: Action tensor of shape (batch, action_dim)
        """
        with torch.no_grad():
            action, self._hidden = self.forward(obs, self._hidden)
        return action
    
    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """Get current hidden state."""
        return self._hidden
    
    def set_hidden_state(self, hidden: torch.Tensor) -> None:
        """Set hidden state."""
        self._hidden = hidden
```

#### المخرجات:
- [x] dense.py - طبقة Dense
- [x] gru.py - طبقة GRU
- [x] policy.py - نموذج RaptorPolicy

---

### اليوم 5: كتابة الاختبارات

#### المهام:
- [ ] اختبارات طبقة Dense
- [ ] اختبارات طبقة GRU
- [ ] اختبارات RaptorPolicy
- [ ] اختبار التوافق مع C++

#### التفاصيل:

**1.7 tests/test_nn.py:**
```python
"""Tests for neural network layers."""
import pytest
import torch
from raptor.nn.dense import Dense
from raptor.nn.gru import GRULayer
from raptor.nn.policy import RaptorPolicy

class TestDense:
    def test_forward_shape(self):
        layer = Dense(22, 16, activation="relu")
        x = torch.randn(8, 22)
        y = layer(x)
        assert y.shape == (8, 16)
    
    def test_relu_activation(self):
        layer = Dense(10, 10, activation="relu")
        x = torch.randn(1, 10)
        y = layer(x)
        assert (y >= 0).all() or (x <= 0).any()
    
    def test_identity_activation(self):
        layer = Dense(10, 10, activation="identity")
        # With identity, output should be linear transformation
        x = torch.ones(1, 10)
        y = layer(x)
        assert y.shape == (1, 10)

class TestGRU:
    def test_forward_shape(self):
        gru = GRULayer(16, 16)
        x = torch.randn(8, 5, 16)  # batch=8, seq=5, features=16
        output, hidden = gru(x)
        assert output.shape == (8, 5, 16)
        assert hidden.shape == (1, 8, 16)
    
    def test_hidden_state_persistence(self):
        gru = GRULayer(16, 16)
        x = torch.randn(1, 1, 16)
        _, hidden1 = gru(x)
        _, hidden2 = gru(x, hidden1)
        # Hidden states should be different
        assert not torch.allclose(hidden1, hidden2)
    
    def test_init_hidden(self):
        gru = GRULayer(16, 16)
        hidden = gru.init_hidden(4, torch.device("cpu"))
        assert hidden.shape == (1, 4, 16)
        assert (hidden == 0).all()

class TestRaptorPolicy:
    def test_forward_shape(self):
        policy = RaptorPolicy(obs_dim=22, hidden_dim=16, action_dim=4)
        obs = torch.randn(8, 22)
        action, hidden = policy(obs)
        assert action.shape == (8, 4)
        assert hidden.shape == (1, 8, 16)
    
    def test_evaluate_step(self):
        policy = RaptorPolicy()
        policy.reset()
        obs = torch.randn(1, 22)
        action = policy.evaluate_step(obs)
        assert action.shape == (1, 4)
    
    def test_hidden_state_management(self):
        policy = RaptorPolicy()
        policy.reset()
        
        # First step
        obs1 = torch.randn(1, 22)
        action1 = policy.evaluate_step(obs1)
        hidden1 = policy.get_hidden_state().clone()
        
        # Second step
        obs2 = torch.randn(1, 22)
        action2 = policy.evaluate_step(obs2)
        hidden2 = policy.get_hidden_state()
        
        # Hidden states should be different
        assert not torch.allclose(hidden1, hidden2)
    
    def test_action_range(self):
        """Actions should be unbounded (identity activation)."""
        policy = RaptorPolicy()
        policy.reset()
        obs = torch.randn(100, 22) * 10  # Large inputs
        action = policy.evaluate_step(obs)
        # Actions can be any value (no tanh/sigmoid)
        assert action.shape == (100, 4)
```

#### المخرجات:
- [x] tests/test_nn.py - اختبارات الشبكة العصبية

---

## الأسبوع 2: الأدوات المساعدة والتكوين

### اليوم 1-2: نظام التكوين

#### المهام:
- [ ] إنشاء ملفات التكوين YAML
- [ ] تنفيذ محمل التكوين
- [ ] تنفيذ ملفات JSON للطائرات

#### التفاصيل:

**1.8 configs/default.yaml:**
```yaml
# Raptor Foundation Policy Configuration

# Environment
environment:
  name: "l2f"
  dt: 0.01  # 100 Hz
  episode_steps: 500  # 5 seconds

# Observation
observation:
  dim: 22
  components:
    position: 3
    rotation_matrix: 9
    linear_velocity: 3
    angular_velocity: 3
    previous_action: 4

# Action
action:
  dim: 4
  range: [-1, 1]

# Network
network:
  hidden_dim: 16
  activation: "relu"

# Training
training:
  device: "cuda"
  seed: 42
```

**1.9 configs/pre_training.yaml:**
```yaml
# Pre-Training Configuration

inherit: default.yaml

# SAC Algorithm
sac:
  actor_batch_size: 128
  critic_batch_size: 128
  gamma: 0.99
  tau: 0.005
  target_entropy: -2.0
  learning_rate: 0.0003
  actor_hidden_dim: 64
  critic_hidden_dim: 256
  n_warmup_steps: 10000

# Training
training:
  num_teachers: 1000
  steps_per_teacher: 1000000
  eval_frequency: 10000
  checkpoint_frequency: 100000
  
# Domain Randomization
domain_randomization:
  thrust_to_weight: [1.5, 5.0]
  torque_to_inertia: [40, 1200]
  mass: [0.02, 5.0]
  rotor_time_constant_rising: [0.03, 0.30]
  rotor_time_constant_falling: [0.07, 0.30]
  rotor_torque_constant: [0.005, 0.05]
  disturbance_force: [0, 0.3]
```

**1.10 configs/post_training.yaml:**
```yaml
# Post-Training Configuration

inherit: default.yaml

# Knowledge Distillation
distillation:
  num_teachers: 1000
  sequence_length: 500
  batch_size: 64
  n_epochs: 1000
  learning_rate: 0.0001
  epoch_teacher_forcing: 10
  solved_return: 300

# Student Network
student:
  hidden_dim: 16
  
# Data Collection
data_collection:
  num_episodes: 10
  num_episodes_eval: 100
  on_policy: true
  shuffle: true
```

**1.11 configs/drones/crazyflie.json:**
```json
{
    "name": "Crazyflie",
    "dynamics": {
        "rotor_positions": [
            [0.028, -0.028, 0.0],
            [-0.028, -0.028, 0.0],
            [-0.028, 0.028, 0.0],
            [0.028, 0.028, 0.0]
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
        "mass": 0.027,
        "gravity": [0.0, 0.0, -9.81],
        "J": [
            [1.65e-5, 0.0, 0.0],
            [0.0, 1.65e-5, 0.0],
            [0.0, 0.0, 2.92e-5]
        ],
        "rotor_torque_constants": [0.005, 0.005, 0.005, 0.005],
        "rotor_time_constants_rising": [0.05, 0.05, 0.05, 0.05],
        "rotor_time_constants_falling": [0.07, 0.07, 0.07, 0.07]
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
    }
}
```

**1.12 src/raptor/utils/config.py:**
```python
"""Configuration management."""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Config:
    """Configuration container."""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __getattr__(self, name: str) -> Any:
        if name in self.data:
            value = self.data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        return self.data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(path, 'r') as f:
        return json.load(f)

def load_config(
    path: Path,
    base_dir: Optional[Path] = None
) -> Config:
    """
    Load configuration with inheritance support.
    
    Args:
        path: Path to configuration file
        base_dir: Base directory for relative paths
    
    Returns:
        Config object
    """
    if base_dir is None:
        base_dir = path.parent
    
    if path.suffix == '.yaml' or path.suffix == '.yml':
        data = load_yaml(path)
    elif path.suffix == '.json':
        data = load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    # Handle inheritance
    if 'inherit' in data:
        parent_path = base_dir / data.pop('inherit')
        parent_config = load_config(parent_path, base_dir)
        # Merge parent with current (current overrides parent)
        merged = {**parent_config.data, **data}
        data = merged
    
    return Config(data)

def save_config(config: Config, path: Path) -> None:
    """Save configuration to file."""
    if path.suffix == '.yaml' or path.suffix == '.yml':
        with open(path, 'w') as f:
            yaml.dump(config.data, f, default_flow_style=False)
    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config.data, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
```

#### المخرجات:
- [x] configs/default.yaml
- [x] configs/pre_training.yaml
- [x] configs/post_training.yaml
- [x] configs/drones/crazyflie.json
- [x] src/raptor/utils/config.py

---

### اليوم 3-4: أدوات Checkpoint والتسجيل

#### المهام:
- [ ] تنفيذ نظام Checkpoint
- [ ] تنفيذ نظام التسجيل (Logging)
- [ ] تكامل TensorBoard

#### التفاصيل:

**1.13 src/raptor/utils/checkpoint.py:**
```python
"""Checkpoint management."""
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class CheckpointManager:
    """Manages saving and loading of checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: list = []
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Path:
        """Save checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'timestamp': timestamp
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        self.checkpoints.append(path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Remove old checkpoints
        self._cleanup()
        
        return path
    
    def load(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return self.load(best_path, model, optimizer)
    
    def _cleanup(self) -> None:
        """Remove old checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
```

**1.14 src/raptor/utils/logging.py:**
```python
"""Logging utilities."""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Logger:
    """Combined console and TensorBoard logger."""
    
    def __init__(
        self,
        log_dir: Path,
        name: str = "raptor",
        level: int = logging.INFO
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            self.log_dir / f"{name}_{timestamp}.log"
        )
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")
        
        self.step = 0
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """Log scalar to TensorBoard."""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log multiple scalars to TensorBoard."""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: Optional[int] = None
    ) -> None:
        """Log histogram to TensorBoard."""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def set_step(self, step: int) -> None:
        """Set current step."""
        self.step = step
    
    def close(self) -> None:
        """Close logger."""
        self.writer.close()
```

#### المخرجات:
- [x] src/raptor/utils/checkpoint.py
- [x] src/raptor/utils/logging.py

---

### اليوم 5: التكامل والاختبار

#### المهام:
- [ ] تكامل جميع المكونات
- [ ] اختبار شامل
- [ ] توثيق المرحلة الأولى

#### التفاصيل:

**1.15 src/raptor/__init__.py:**
```python
"""Raptor Foundation Policy for Quadrotor Control."""
from .nn.policy import RaptorPolicy
from .utils.config import Config, load_config
from .utils.checkpoint import CheckpointManager
from .utils.logging import Logger

__version__ = "0.1.0"
__all__ = [
    "RaptorPolicy",
    "Config",
    "load_config",
    "CheckpointManager",
    "Logger",
]
```

**1.16 تشغيل الاختبارات:**
```bash
pytest tests/ -v --cov=src/raptor
```

#### معايير النجاح للمرحلة 1:
- [ ] جميع الاختبارات تمر
- [ ] تغطية الكود > 80%
- [ ] الشبكة العصبية تعمل بشكل صحيح
- [ ] نظام التكوين يعمل
- [ ] نظام Checkpoint يعمل
- [ ] TensorBoard يعمل

---

