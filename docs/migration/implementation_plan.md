# Raptor Foundation Policy Migration - Implementation Plan
# خطة تنفيذية تفصيلية لهجرة سياسة Raptor الأساسية

**تاريخ الإنشاء:** 2 فبراير 2026  
**المدة الإجمالية:** 10 أسابيع  
**الهدف:** هجرة مشروع Raptor من C++ إلى Python/PyTorch  

---

# نظرة عامة على المراحل

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         الجدول الزمني العام                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  الأسبوع 1-2    │  الأسبوع 3-4    │  الأسبوع 5-6    │  الأسبوع 7-8    │ 9-10 │
│  ┌──────────┐   │  ┌──────────┐   │  ┌──────────┐   │  ┌──────────┐   │ ┌──┐ │
│  │ المرحلة 1│   │  │ المرحلة 2│   │  │ المرحلة 3│   │  │ المرحلة 4│   │ │5-6│ │
│  │ الأساس   │──►│  │ البنية   │──►│  │ التدريب  │──►│  │ التقطير  │──►│ │  │ │
│  │          │   │  │ التحتية  │   │  │ المسبق   │   │  │          │   │ └──┘ │
│  └──────────┘   │  └──────────┘   │  └──────────┘   │  └──────────┘   │      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| المرحلة | الاسم | المدة | الأسابيع |
|---------|-------|-------|----------|
| 1 | الأساس وهيكل المشروع | أسبوعان | 1-2 |
| 2 | البنية التحتية للتعلم المعزز | أسبوعان | 3-4 |
| 3 | التدريب المسبق (Pre-Training) | أسبوعان | 5-6 |
| 4 | التدريب اللاحق (Post-Training) | أسبوعان | 7-8 |
| 5 | التصدير والنشر | أسبوع | 9 |
| 6 | الاختبار والتوثيق | أسبوع | 10 |

---

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

# المرحلة 5: التصدير والنشر
## Phase 5: Export & Deployment

**المدة:** أسبوع واحد (الأسبوع 9)  
**الهدف:** تصدير النموذج إلى ONNX و TFLite

---

## الأسبوع 9: التصدير

### اليوم 1-2: تصدير ONNX

#### المهام:
- [ ] تنفيذ ONNX Exporter
- [ ] التحقق من صحة التصدير
- [ ] اختبار الاستدلال

#### التفاصيل:

**5.1 src/raptor/export/onnx_export.py:**
```python
"""ONNX export utilities."""
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Tuple

from ..nn.policy import RaptorPolicy

def export_to_onnx(
    model: RaptorPolicy,
    output_path: Path,
    opset_version: int = 14
) -> None:
    """Export model to ONNX format."""
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    obs_dim = model.obs_dim
    hidden_dim = model.hidden_dim
    
    dummy_obs = torch.randn(batch_size, obs_dim)
    dummy_hidden = torch.zeros(1, batch_size, hidden_dim)
    
    # Export
    torch.onnx.export(
        model,
        (dummy_obs, dummy_hidden),
        str(output_path),
        input_names=['observation', 'hidden_in'],
        output_names=['action', 'hidden_out'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'hidden_in': {1: 'batch_size'},
            'action': {0: 'batch_size'},
            'hidden_out': {1: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    # Verify
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {output_path}")

def verify_onnx(
    pytorch_model: RaptorPolicy,
    onnx_path: Path,
    num_tests: int = 10
) -> bool:
    """Verify ONNX model matches PyTorch model."""
    pytorch_model.eval()
    
    # Create ONNX runtime session
    session = ort.InferenceSession(str(onnx_path))
    
    for _ in range(num_tests):
        # Random inputs
        obs = np.random.randn(1, pytorch_model.obs_dim).astype(np.float32)
        hidden = np.zeros((1, 1, pytorch_model.hidden_dim), dtype=np.float32)
        
        # PyTorch inference
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            hidden_tensor = torch.FloatTensor(hidden)
            pt_action, pt_hidden = pytorch_model(obs_tensor, hidden_tensor)
            pt_action = pt_action.numpy()
            pt_hidden = pt_hidden.numpy()
        
        # ONNX inference
        onnx_outputs = session.run(
            None,
            {'observation': obs, 'hidden_in': hidden}
        )
        onnx_action, onnx_hidden = onnx_outputs
        
        # Compare
        if not np.allclose(pt_action, onnx_action, rtol=1e-4, atol=1e-4):
            print(f"Action mismatch: PyTorch={pt_action}, ONNX={onnx_action}")
            return False
        
        if not np.allclose(pt_hidden, onnx_hidden, rtol=1e-4, atol=1e-4):
            print(f"Hidden mismatch: PyTorch={pt_hidden}, ONNX={onnx_hidden}")
            return False
    
    print("ONNX verification passed!")
    return True
```

#### المخرجات:
- [x] src/raptor/export/onnx_export.py

---

### اليوم 3-4: تحويل TFLite

#### المهام:
- [ ] تنفيذ TFLite Converter
- [ ] التحقق من صحة التحويل
- [ ] اختبار على Android (اختياري)

#### التفاصيل:

**5.2 src/raptor/export/tflite_convert.py:**
```python
"""TFLite conversion utilities."""
import subprocess
from pathlib import Path
from typing import Optional

def convert_onnx_to_tflite(
    onnx_path: Path,
    tflite_path: Path,
    quantize: bool = False
) -> None:
    """Convert ONNX model to TFLite."""
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Export to SavedModel
    saved_model_dir = onnx_path.parent / "tf_saved_model"
    tf_rep.export_graph(str(saved_model_dir))
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

def verify_tflite(
    tflite_path: Path,
    onnx_path: Path,
    num_tests: int = 10
) -> bool:
    """Verify TFLite model matches ONNX model."""
    import tensorflow as tf
    import onnxruntime as ort
    import numpy as np
    
    # Load TFLite
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load ONNX
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    for _ in range(num_tests):
        # Random inputs
        obs = np.random.randn(1, 22).astype(np.float32)
        hidden = np.zeros((1, 1, 16), dtype=np.float32)
        
        # ONNX inference
        onnx_outputs = onnx_session.run(
            None,
            {'observation': obs, 'hidden_in': hidden}
        )
        
        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], obs)
        interpreter.set_tensor(input_details[1]['index'], hidden)
        interpreter.invoke()
        
        tflite_action = interpreter.get_tensor(output_details[0]['index'])
        
        # Compare
        if not np.allclose(onnx_outputs[0], tflite_action, rtol=1e-3, atol=1e-3):
            print(f"Mismatch: ONNX={onnx_outputs[0]}, TFLite={tflite_action}")
            return False
    
    print("TFLite verification passed!")
    return True
```

**5.3 scripts/export_model.py:**
```python
#!/usr/bin/env python
"""Script to export trained model."""
import argparse
from pathlib import Path
import torch

from raptor.nn.policy import RaptorPolicy
from raptor.export.onnx_export import export_to_onnx, verify_onnx
from raptor.export.tflite_convert import convert_onnx_to_tflite, verify_tflite

def main():
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="exports")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = RaptorPolicy()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    
    # Export to ONNX
    onnx_path = output_dir / "raptor_policy.onnx"
    export_to_onnx(model, onnx_path)
    
    # Verify ONNX
    assert verify_onnx(model, onnx_path), "ONNX verification failed!"
    
    # Convert to TFLite
    tflite_path = output_dir / "raptor_policy.tflite"
    convert_onnx_to_tflite(onnx_path, tflite_path, quantize=args.quantize)
    
    # Verify TFLite
    assert verify_tflite(tflite_path, onnx_path), "TFLite verification failed!"
    
    print(f"\nExport complete!")
    print(f"ONNX: {onnx_path}")
    print(f"TFLite: {tflite_path}")

if __name__ == "__main__":
    main()
```

#### المخرجات:
- [x] src/raptor/export/tflite_convert.py
- [x] scripts/export_model.py

---

### اليوم 5: توثيق التصدير

#### المهام:
- [ ] كتابة دليل التصدير
- [ ] كتابة مثال Android
- [ ] اختبار نهائي

#### معايير النجاح للمرحلة 5:
- [ ] ملف ONNX صالح
- [ ] ملف TFLite صالح
- [ ] التحقق من التوافق
- [ ] حجم النموذج < 100 KB

---

# المرحلة 6: الاختبار والتوثيق
## Phase 6: Testing & Documentation

**المدة:** أسبوع واحد (الأسبوع 10)  
**الهدف:** اختبار شامل وتوثيق كامل

---

## الأسبوع 10: الاختبار والتوثيق

### اليوم 1-2: اختبارات شاملة

#### المهام:
- [ ] اختبارات الوحدة (Unit Tests)
- [ ] اختبارات التكامل (Integration Tests)
- [ ] اختبارات الأداء (Performance Tests)

### اليوم 3-4: التوثيق

#### المهام:
- [ ] README.md شامل
- [ ] توثيق API
- [ ] أمثلة الاستخدام
- [ ] دليل المساهمة

### اليوم 5: المراجعة النهائية

#### المهام:
- [ ] مراجعة الكود
- [ ] تنظيف الملفات
- [ ] إنشاء Release
- [ ] نشر على PyPI (اختياري)

#### معايير النجاح للمرحلة 6:
- [ ] تغطية الاختبارات > 80%
- [ ] جميع الاختبارات تمر
- [ ] توثيق كامل
- [ ] README واضح

---

# المكونات الإضافية للنشر الواقعي
## Additional Components for Real-World Deployment

هذا القسم يحتوي على المكونات الإضافية الضرورية لضمان أن السياسة المدربة جاهزة للتنفيذ على طائرة حقيقية.

---

## 1. تتبع المسار (Trajectory Tracking - Langevin)

### 1.1 الأهمية
وفقاً لما اتفقنا عليه، التدريب يجب أن يشمل:
- **50% مهمة التحويم (Hover)**
- **50% مهمة تتبع المسار (Trajectory Tracking)**

### 1.2 تنفيذ Langevin Trajectory Generator

**src/raptor/environments/trajectory.py:**
```python
"""Trajectory generation for training."""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class LangevinParams:
    """Langevin trajectory parameters."""
    gamma: float = 1.0    # Damping coefficient
    omega: float = 2.0    # Frequency
    sigma: float = 0.5    # Noise scale
    alpha: float = 0.01   # Smoothing factor
    dt: float = 0.01      # Time step

class LangevinTrajectory:
    """
    Langevin dynamics trajectory generator.
    
    Generates smooth, random trajectories for trajectory tracking training.
    """
    
    def __init__(self, params: Optional[LangevinParams] = None):
        self.params = params or LangevinParams()
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.rng = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset trajectory to origin."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        return self.position.copy()
    
    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate next trajectory point using Langevin dynamics.
        
        Returns:
            position: Target position (3,)
            velocity: Target velocity (3,)
        """
        p = self.params
        
        # Langevin dynamics: dv = -gamma*v*dt - omega^2*x*dt + sigma*dW
        noise = self.rng.normal(0, 1, 3) * np.sqrt(p.dt)
        
        # Update velocity
        acceleration = (
            -p.gamma * self.velocity 
            - p.omega**2 * self.position 
            + p.sigma * noise / p.dt
        )
        self.velocity += acceleration * p.dt
        
        # Update position
        self.position += self.velocity * p.dt
        
        # Clamp position to reasonable bounds
        self.position = np.clip(self.position, -1.0, 1.0)
        
        return self.position.copy(), self.velocity.copy()
    
    def get_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current target position and velocity."""
        return self.position.copy(), self.velocity.copy()


class TrajectoryManager:
    """
    Manages trajectory selection for training.
    
    Randomly selects between hover and trajectory tracking tasks.
    """
    
    def __init__(
        self,
        hover_probability: float = 0.5,
        langevin_params: Optional[LangevinParams] = None
    ):
        self.hover_probability = hover_probability
        self.langevin = LangevinTrajectory(langevin_params)
        self.is_hover = True
        self.rng = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset and randomly select task type.
        
        Returns:
            target_position: Initial target position
            target_velocity: Initial target velocity
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Randomly select task type
        self.is_hover = self.rng.random() < self.hover_probability
        
        if self.is_hover:
            return np.zeros(3), np.zeros(3)
        else:
            return self.langevin.reset(seed)
    
    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get next target.
        
        Returns:
            target_position: Target position
            target_velocity: Target velocity
        """
        if self.is_hover:
            return np.zeros(3), np.zeros(3)
        else:
            return self.langevin.step()
    
    def get_task_type(self) -> str:
        """Get current task type."""
        return "hover" if self.is_hover else "trajectory"
```

### 1.3 تحديث L2F Wrapper لدعم Trajectory Tracking

**تحديث src/raptor/environments/l2f_wrapper.py:**
```python
# إضافة في __init__:
from .trajectory import TrajectoryManager, LangevinParams

class L2FEnv(gym.Env):
    def __init__(
        self,
        drone_config: Optional[str] = None,
        dt: float = 0.01,
        episode_steps: int = 500,
        render_mode: Optional[str] = None,
        seed: int = 0,
        # إضافات جديدة
        hover_probability: float = 0.5,
        observation_noise: Optional[Dict[str, float]] = None,
        action_noise: float = 0.0
    ):
        # ... الكود الموجود ...
        
        # Trajectory manager
        self.trajectory_manager = TrajectoryManager(
            hover_probability=hover_probability,
            langevin_params=LangevinParams(dt=dt)
        )
        
        # Noise parameters
        self.observation_noise = observation_noise or {
            'position': 0.01,
            'orientation': 0.01,
            'linear_velocity': 0.05,
            'angular_velocity': 0.1
        }
        self.action_noise = action_noise
        
        # Current target
        self._target_position = np.zeros(3)
        self._target_velocity = np.zeros(3)
    
    def reset(self, seed=None, options=None):
        # ... الكود الموجود ...
        
        # Reset trajectory and get initial target
        self._target_position, self._target_velocity = self.trajectory_manager.reset(seed)
        
        return self._observation.copy(), {
            'task_type': self.trajectory_manager.get_task_type()
        }
    
    def step(self, action):
        # Add action noise
        if self.action_noise > 0:
            action = action + np.random.normal(0, self.action_noise, action.shape)
            action = np.clip(action, -1.0, 1.0)
        
        # ... الكود الموجود ...
        
        # Update target
        self._target_position, self._target_velocity = self.trajectory_manager.step()
        
        # ... الكود الموجود ...
    
    def _get_observation(self):
        # ... الكود الموجود ...
        
        # Add observation noise
        if self.observation_noise:
            noise = np.zeros(22, dtype=np.float32)
            noise[0:3] = np.random.normal(0, self.observation_noise['position'], 3)
            noise[3:12] = np.random.normal(0, self.observation_noise['orientation'], 9)
            noise[12:15] = np.random.normal(0, self.observation_noise['linear_velocity'], 3)
            noise[15:18] = np.random.normal(0, self.observation_noise['angular_velocity'], 3)
            self._observation += noise
    
    def _compute_reward(self):
        # Position error relative to target
        pos = self._observation[:3]
        pos_error = np.sum((pos - self._target_position) ** 2)
        
        # ... بقية حساب المكافأة ...
```

---

## 2. العرض ثلاثي الأبعاد (3D Visualization via WebSocket)

### 2.1 تنفيذ WebSocket UI Server

**src/raptor/visualization/ui_server.py:**
```python
"""WebSocket UI server for 3D visualization."""
import asyncio
import json
import websockets
from typing import Optional, Dict, Any
import numpy as np

class UIServer:
    """
    WebSocket server for real-time 3D visualization.
    
    Connects to ui-server package for rendering.
    Default URL: http://localhost:13337
    """
    
    def __init__(self, host: str = "localhost", port: int = 13337):
        self.host = host
        self.port = port
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to UI server."""
        try:
            self.websocket = await websockets.connect(
                f"ws://{self.host}:{self.port}/ws"
            )
            self.connected = True
            print(f"Connected to UI server at ws://{self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to UI server: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from UI server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
    
    async def send_state(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        target_position: Optional[np.ndarray] = None,
        motor_commands: Optional[np.ndarray] = None
    ) -> None:
        """
        Send drone state to UI server.
        
        Args:
            position: Drone position (3,)
            rotation: Rotation matrix (3, 3) or quaternion (4,)
            target_position: Target position for visualization
            motor_commands: Motor commands (4,)
        """
        if not self.connected or not self.websocket:
            return
        
        # Convert rotation matrix to quaternion if needed
        if rotation.shape == (3, 3):
            quat = self._rotation_matrix_to_quaternion(rotation)
        else:
            quat = rotation
        
        message = {
            "type": "drone_state",
            "position": position.tolist(),
            "quaternion": quat.tolist(),
        }
        
        if target_position is not None:
            message["target_position"] = target_position.tolist()
        
        if motor_commands is not None:
            message["motor_commands"] = motor_commands.tolist()
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            print(f"Failed to send state: {e}")
            self.connected = False
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])


class SyncUIServer:
    """Synchronous wrapper for UIServer."""
    
    def __init__(self, host: str = "localhost", port: int = 13337):
        self.server = UIServer(host, port)
        self.loop = asyncio.new_event_loop()
    
    def connect(self) -> bool:
        return self.loop.run_until_complete(self.server.connect())
    
    def disconnect(self) -> None:
        self.loop.run_until_complete(self.server.disconnect())
    
    def send_state(self, **kwargs) -> None:
        self.loop.run_until_complete(self.server.send_state(**kwargs))
```

### 2.2 استخدام العرض ثلاثي الأبعاد

```bash
# تثبيت ui-server
pip install ui-server==0.0.13

# تشغيل الخادم
ui-server

# فتح المتصفح على http://localhost:13337
```

---

## 3. وضع الاستدلال بمعدل 250 Hz

### 3.1 تنفيذ High-Frequency Inference

**src/raptor/inference/realtime.py:**
```python
"""Real-time inference at 250 Hz."""
import torch
import numpy as np
import time
from typing import Optional, Tuple
from ..nn.policy import RaptorPolicy

class RealtimeInference:
    """
    Real-time policy inference at configurable frequency.
    
    Supports frequency decoupling: policy trained at 100 Hz
    can run at higher frequencies (250 Hz, 400 Hz).
    """
    
    def __init__(
        self,
        model: RaptorPolicy,
        target_frequency: float = 250.0,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.target_frequency = target_frequency
        self.target_dt = 1.0 / target_frequency
        
        # Hidden state
        self.hidden = None
        self.reset()
        
        # Timing
        self.last_time = None
        self.inference_times = []
    
    def reset(self) -> None:
        """Reset hidden state."""
        self.hidden = torch.zeros(1, 1, self.model.hidden_dim, device=self.device)
        self.last_time = None
        self.inference_times = []
    
    @torch.no_grad()
    def get_action(
        self,
        observation: np.ndarray,
        wait_for_timing: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Get action from policy.
        
        Args:
            observation: Current observation (22,)
            wait_for_timing: If True, wait to maintain target frequency
        
        Returns:
            action: Motor commands (4,)
            info: Timing information
        """
        start_time = time.perf_counter()
        
        # Wait for timing if needed
        if wait_for_timing and self.last_time is not None:
            elapsed = start_time - self.last_time
            if elapsed < self.target_dt:
                time.sleep(self.target_dt - elapsed)
                start_time = time.perf_counter()
        
        # Convert observation
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Forward pass
        action, self.hidden = self.model(obs_tensor, self.hidden)
        action = action.cpu().numpy().squeeze()
        
        # Timing
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        self.last_time = start_time
        
        info = {
            'inference_time_ms': inference_time * 1000,
            'target_dt_ms': self.target_dt * 1000,
            'actual_frequency': 1.0 / inference_time if inference_time > 0 else float('inf')
        }
        
        return action, info
    
    def get_timing_stats(self) -> dict:
        """Get timing statistics."""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'mean_inference_ms': np.mean(times) * 1000,
            'std_inference_ms': np.std(times) * 1000,
            'max_inference_ms': np.max(times) * 1000,
            'min_inference_ms': np.min(times) * 1000,
            'mean_frequency': 1.0 / np.mean(times),
            'meets_target': np.mean(times) < self.target_dt
        }


def benchmark_inference(
    model: RaptorPolicy,
    num_iterations: int = 1000,
    device: str = "cpu"
) -> dict:
    """
    Benchmark inference latency.
    
    Args:
        model: Policy model
        num_iterations: Number of iterations
        device: Device to run on
    
    Returns:
        Benchmark results
    """
    inference = RealtimeInference(model, device=device)
    
    # Warmup
    obs = np.random.randn(22).astype(np.float32)
    for _ in range(100):
        inference.get_action(obs, wait_for_timing=False)
    
    # Benchmark
    inference.reset()
    for _ in range(num_iterations):
        obs = np.random.randn(22).astype(np.float32)
        inference.get_action(obs, wait_for_timing=False)
    
    stats = inference.get_timing_stats()
    
    print(f"Inference Benchmark Results ({device}):")
    print(f"  Mean: {stats['mean_inference_ms']:.3f} ms")
    print(f"  Std:  {stats['std_inference_ms']:.3f} ms")
    print(f"  Max:  {stats['max_inference_ms']:.3f} ms")
    print(f"  Frequency: {stats['mean_frequency']:.1f} Hz")
    print(f"  Meets 250 Hz: {stats['meets_target']}")
    
    return stats
```

---

## 4. دليل التكامل مع الطائرة الحقيقية

### 4.1 متطلبات الأجهزة

| المكون | الوصف | المتطلبات |
|--------|-------|-----------|
| **IMU** | وحدة القياس بالقصور الذاتي | 200+ Hz, 6-axis minimum |
| **Position Sensor** | GPS/Motion Capture/VIO | 50+ Hz |
| **Flight Controller** | متحكم الطيران | PWM output, 250+ Hz |
| **Companion Computer** | حاسوب مصاحب | Android phone / Raspberry Pi |
| **Communication** | اتصال | WiFi/Bluetooth/Serial |

### 4.2 تحويل الإحداثيات (Coordinate Transformation)

**src/raptor/realworld/coordinate_transform.py:**
```python
"""Coordinate transformation utilities."""
import numpy as np
from typing import Tuple

# FLU Convention: x = forward, y = left, z = up
# NED Convention: x = north, y = east, z = down

def ned_to_flu(position: np.ndarray, velocity: np.ndarray = None) -> Tuple[np.ndarray, ...]:
    """
    Convert NED (North-East-Down) to FLU (Forward-Left-Up).
    
    Args:
        position: Position in NED frame (3,)
        velocity: Velocity in NED frame (3,) optional
    
    Returns:
        Position (and velocity) in FLU frame
    """
    # Rotation matrix from NED to FLU
    R_ned_to_flu = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    pos_flu = R_ned_to_flu @ position
    
    if velocity is not None:
        vel_flu = R_ned_to_flu @ velocity
        return pos_flu, vel_flu
    
    return pos_flu,


def enu_to_flu(position: np.ndarray, velocity: np.ndarray = None) -> Tuple[np.ndarray, ...]:
    """
    Convert ENU (East-North-Up) to FLU (Forward-Left-Up).
    
    Args:
        position: Position in ENU frame (3,)
        velocity: Velocity in ENU frame (3,) optional
    
    Returns:
        Position (and velocity) in FLU frame
    """
    # Rotation matrix from ENU to FLU (assuming forward = north)
    R_enu_to_flu = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    
    pos_flu = R_enu_to_flu @ position
    
    if velocity is not None:
        vel_flu = R_enu_to_flu @ velocity
        return pos_flu, vel_flu
    
    return pos_flu,


def motor_commands_to_pwm(
    normalized_commands: np.ndarray,
    min_pwm: int = 1000,
    max_pwm: int = 2000
) -> np.ndarray:
    """
    Convert normalized motor commands [-1, 1] to PWM values.
    
    Args:
        normalized_commands: Motor commands in [-1, 1] range (4,)
        min_pwm: Minimum PWM value (typically 1000)
        max_pwm: Maximum PWM value (typically 2000)
    
    Returns:
        PWM values (4,)
    """
    # Map [-1, 1] to [min_pwm, max_pwm]
    pwm = (normalized_commands + 1) / 2 * (max_pwm - min_pwm) + min_pwm
    return pwm.astype(np.int32)


# Motor ordering verification
MOTOR_ORDER = {
    'front_right': 0,
    'back_right': 1,
    'back_left': 2,
    'front_left': 3
}

def verify_motor_order(
    test_commands: np.ndarray,
    expected_response: str
) -> bool:
    """
    Helper to verify motor ordering matches physical drone.
    
    Usage:
        1. Send [1, 0, 0, 0] - front-right should spin
        2. Send [0, 1, 0, 0] - back-right should spin
        3. Send [0, 0, 1, 0] - back-left should spin
        4. Send [0, 0, 0, 1] - front-left should spin
    """
    print(f"Testing motor command: {test_commands}")
    print(f"Expected: {expected_response}")
    return True  # Manual verification required
```

### 4.3 فحوصات السلامة (Safety Checks)

**src/raptor/realworld/safety.py:**
```python
"""Safety checks for real-world deployment."""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class SafetyLimits:
    """Safety limits for real-world operation."""
    max_position: float = 5.0          # meters from origin
    max_velocity: float = 5.0          # m/s
    max_angular_velocity: float = 10.0  # rad/s
    max_tilt_angle: float = 1.0        # radians (~57 degrees)
    min_altitude: float = 0.1          # meters
    max_altitude: float = 10.0         # meters
    max_action_change: float = 0.5     # per step
    emergency_stop_threshold: float = 0.8  # of any limit

class SafetyMonitor:
    """
    Safety monitor for real-world drone operation.
    
    Monitors state and actions, triggers emergency stop if limits exceeded.
    """
    
    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        self.prev_action = np.zeros(4)
        self.emergency_stop = False
        self.violation_count = 0
        self.violation_history = []
    
    def check_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
        rotation_matrix: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if state is within safe limits.
        
        Returns:
            is_safe: True if state is safe
            message: Description of any violations
        """
        violations = []
        
        # Position check
        pos_norm = np.linalg.norm(position[:2])  # horizontal distance
        if pos_norm > self.limits.max_position:
            violations.append(f"Position exceeded: {pos_norm:.2f}m > {self.limits.max_position}m")
        
        # Altitude check
        altitude = position[2]
        if altitude < self.limits.min_altitude:
            violations.append(f"Altitude too low: {altitude:.2f}m < {self.limits.min_altitude}m")
        if altitude > self.limits.max_altitude:
            violations.append(f"Altitude too high: {altitude:.2f}m > {self.limits.max_altitude}m")
        
        # Velocity check
        vel_norm = np.linalg.norm(velocity)
        if vel_norm > self.limits.max_velocity:
            violations.append(f"Velocity exceeded: {vel_norm:.2f}m/s > {self.limits.max_velocity}m/s")
        
        # Angular velocity check
        ang_vel_norm = np.linalg.norm(angular_velocity)
        if ang_vel_norm > self.limits.max_angular_velocity:
            violations.append(f"Angular velocity exceeded: {ang_vel_norm:.2f}rad/s")
        
        # Tilt angle check (from rotation matrix)
        tilt_angle = np.arccos(np.clip(rotation_matrix[2, 2], -1, 1))
        if tilt_angle > self.limits.max_tilt_angle:
            violations.append(f"Tilt angle exceeded: {np.degrees(tilt_angle):.1f}° > {np.degrees(self.limits.max_tilt_angle):.1f}°")
        
        if violations:
            self.violation_count += 1
            self.violation_history.extend(violations)
            return False, "; ".join(violations)
        
        return True, "OK"
    
    def check_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Check and potentially limit action.
        
        Returns:
            safe_action: Action within safe limits
            was_modified: True if action was modified
        """
        # Clamp to [-1, 1]
        safe_action = np.clip(action, -1.0, 1.0)
        
        # Limit rate of change
        action_change = safe_action - self.prev_action
        if np.any(np.abs(action_change) > self.limits.max_action_change):
            action_change = np.clip(
                action_change,
                -self.limits.max_action_change,
                self.limits.max_action_change
            )
            safe_action = self.prev_action + action_change
        
        was_modified = not np.allclose(action, safe_action)
        self.prev_action = safe_action.copy()
        
        return safe_action, was_modified
    
    def trigger_emergency_stop(self) -> np.ndarray:
        """
        Trigger emergency stop.
        
        Returns:
            Emergency action (motors off or hover thrust)
        """
        self.emergency_stop = True
        # Return zero thrust (motors off)
        # In practice, you might want to return hover thrust instead
        return np.array([-1.0, -1.0, -1.0, -1.0])
    
    def reset(self) -> None:
        """Reset safety monitor."""
        self.prev_action = np.zeros(4)
        self.emergency_stop = False
        self.violation_count = 0
        self.violation_history = []
```

---

## 5. دليل تكامل Android الكامل

### 5.1 هيكل مشروع Android

```
android-raptor/
├── app/
│   ├── src/main/
│   │   ├── java/com/raptor/drone/
│   │   │   ├── MainActivity.kt
│   │   │   ├── PolicyInference.kt
│   │   │   ├── SensorManager.kt
│   │   │   ├── DroneController.kt
│   │   │   └── SafetyMonitor.kt
│   │   ├── assets/
│   │   │   └── raptor_policy.tflite
│   │   └── res/
│   └── build.gradle
└── build.gradle
```

### 5.2 كود Kotlin الكامل

**PolicyInference.kt:**
```kotlin
package com.raptor.drone

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class PolicyInference(context: Context) {
    private val interpreter: Interpreter
    private val hiddenState: FloatArray = FloatArray(16)
    
    // Input/output buffers
    private val observationBuffer: ByteBuffer
    private val hiddenInputBuffer: ByteBuffer
    private val actionBuffer: ByteBuffer
    private val hiddenOutputBuffer: ByteBuffer
    
    init {
        // Load model
        val modelBuffer = loadModelFile(context, "raptor_policy.tflite")
        interpreter = Interpreter(modelBuffer)
        
        // Initialize buffers
        observationBuffer = ByteBuffer.allocateDirect(22 * 4).order(ByteOrder.nativeOrder())
        hiddenInputBuffer = ByteBuffer.allocateDirect(16 * 4).order(ByteOrder.nativeOrder())
        actionBuffer = ByteBuffer.allocateDirect(4 * 4).order(ByteOrder.nativeOrder())
        hiddenOutputBuffer = ByteBuffer.allocateDirect(16 * 4).order(ByteOrder.nativeOrder())
        
        // Initialize hidden state to zeros
        reset()
    }
    
    fun reset() {
        hiddenState.fill(0f)
    }
    
    fun getAction(observation: FloatArray): FloatArray {
        require(observation.size == 22) { "Observation must have 22 elements" }
        
        // Prepare inputs
        observationBuffer.rewind()
        observation.forEach { observationBuffer.putFloat(it) }
        
        hiddenInputBuffer.rewind()
        hiddenState.forEach { hiddenInputBuffer.putFloat(it) }
        
        // Prepare outputs
        actionBuffer.rewind()
        hiddenOutputBuffer.rewind()
        
        // Run inference
        val inputs = arrayOf(observationBuffer, hiddenInputBuffer)
        val outputs = mapOf(
            0 to actionBuffer,
            1 to hiddenOutputBuffer
        )
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        
        // Extract action
        actionBuffer.rewind()
        val action = FloatArray(4)
        for (i in 0 until 4) {
            action[i] = actionBuffer.float
        }
        
        // Update hidden state
        hiddenOutputBuffer.rewind()
        for (i in 0 until 16) {
            hiddenState[i] = hiddenOutputBuffer.float
        }
        
        return action
    }
    
    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun close() {
        interpreter.close()
    }
}
```

**SensorManager.kt:**
```kotlin
package com.raptor.drone

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager as AndroidSensorManager

class SensorManager(context: Context) : SensorEventListener {
    private val sensorManager: AndroidSensorManager
    private val accelerometer: Sensor?
    private val gyroscope: Sensor?
    
    // Sensor data
    private var acceleration = FloatArray(3)
    private var angularVelocity = FloatArray(3)
    private var rotationMatrix = FloatArray(9)
    
    // Position (from external source like GPS or motion capture)
    var position = FloatArray(3)
    var linearVelocity = FloatArray(3)
    
    // Previous action
    var previousAction = FloatArray(4)
    
    init {
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as AndroidSensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    }
    
    fun start() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, AndroidSensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, AndroidSensorManager.SENSOR_DELAY_FASTEST)
        }
    }
    
    fun stop() {
        sensorManager.unregisterListener(this)
    }
    
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                acceleration = event.values.clone()
            }
            Sensor.TYPE_GYROSCOPE -> {
                angularVelocity = event.values.clone()
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    
    fun getObservation(): FloatArray {
        // Build 22-element observation vector
        val observation = FloatArray(22)
        
        // Position (0:3)
        System.arraycopy(position, 0, observation, 0, 3)
        
        // Rotation matrix (3:12)
        System.arraycopy(rotationMatrix, 0, observation, 3, 9)
        
        // Linear velocity (12:15)
        System.arraycopy(linearVelocity, 0, observation, 12, 3)
        
        // Angular velocity (15:18)
        System.arraycopy(angularVelocity, 0, observation, 15, 3)
        
        // Previous action (18:22)
        System.arraycopy(previousAction, 0, observation, 18, 4)
        
        return observation
    }
}
```

### 5.3 build.gradle Dependencies

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'  // Optional GPU support
}
```

---

## 6. اختبار Sim-to-Real

### 6.1 إجراءات التحقق

**scripts/validate_sim_to_real.py:**
```python
#!/usr/bin/env python
"""Validate policy for sim-to-real transfer."""
import argparse
import numpy as np
import torch
from pathlib import Path

from raptor.nn.policy import RaptorPolicy
from raptor.environments.l2f_wrapper import L2FEnv
from raptor.inference.realtime import benchmark_inference
from raptor.realworld.safety import SafetyMonitor, SafetyLimits

def validate_policy(model_path: str, num_episodes: int = 100):
    """
    Comprehensive validation for sim-to-real transfer.
    """
    print("=" * 60)
    print("Sim-to-Real Validation")
    print("=" * 60)
    
    # Load model
    model = RaptorPolicy()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    results = {}
    
    # 1. Inference latency test
    print("\n1. Inference Latency Test")
    print("-" * 40)
    latency_stats = benchmark_inference(model, num_iterations=1000, device="cpu")
    results['latency'] = latency_stats
    
    if latency_stats['mean_inference_ms'] > 4.0:  # 250 Hz = 4ms
        print("WARNING: Inference too slow for 250 Hz!")
    
    # 2. Domain randomization robustness test
    print("\n2. Domain Randomization Robustness Test")
    print("-" * 40)
    
    success_count = 0
    total_reward = 0
    
    for i in range(num_episodes):
        env = L2FEnv(
            episode_steps=500,
            hover_probability=0.5,
            observation_noise={'position': 0.01, 'orientation': 0.01, 
                              'linear_velocity': 0.05, 'angular_velocity': 0.1},
            action_noise=0.05
        )
        
        obs, _ = env.reset(seed=i)
        hidden = torch.zeros(1, 1, 16)
        episode_reward = 0
        
        for _ in range(500):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, hidden = model(obs_tensor, hidden)
                action = action.numpy().squeeze()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
        if episode_reward > 200:  # Success threshold
            success_count += 1
        
        env.close()
    
    success_rate = success_count / num_episodes
    avg_reward = total_reward / num_episodes
    
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Average reward: {avg_reward:.2f}")
    
    results['robustness'] = {
        'success_rate': success_rate,
        'avg_reward': avg_reward
    }
    
    # 3. Safety bounds test
    print("\n3. Safety Bounds Test")
    print("-" * 40)
    
    safety_monitor = SafetyMonitor(SafetyLimits())
    violations = 0
    
    env = L2FEnv(episode_steps=500)
    obs, _ = env.reset()
    hidden = torch.zeros(1, 1, 16)
    
    for _ in range(500):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, hidden = model(obs_tensor, hidden)
            action = action.numpy().squeeze()
        
        # Check safety
        position = obs[:3]
        velocity = obs[12:15]
        angular_velocity = obs[15:18]
        rotation_matrix = obs[3:12].reshape(3, 3)
        
        is_safe, msg = safety_monitor.check_state(
            position, velocity, angular_velocity, rotation_matrix
        )
        if not is_safe:
            violations += 1
        
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    
    print(f"Safety violations: {violations}")
    results['safety'] = {'violations': violations}
    
    # 4. Action smoothness test
    print("\n4. Action Smoothness Test")
    print("-" * 40)
    
    env = L2FEnv(episode_steps=500)
    obs, _ = env.reset()
    hidden = torch.zeros(1, 1, 16)
    
    actions = []
    for _ in range(500):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, hidden = model(obs_tensor, hidden)
            action = action.numpy().squeeze()
        
        actions.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    
    actions = np.array(actions)
    action_changes = np.diff(actions, axis=0)
    max_change = np.max(np.abs(action_changes))
    mean_change = np.mean(np.abs(action_changes))
    
    print(f"Max action change: {max_change:.4f}")
    print(f"Mean action change: {mean_change:.4f}")
    
    results['smoothness'] = {
        'max_change': max_change,
        'mean_change': mean_change
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    
    if latency_stats['mean_inference_ms'] <= 4.0:
        print("[PASS] Latency: Meets 250 Hz requirement")
    else:
        print("[FAIL] Latency: Too slow for 250 Hz")
        all_passed = False
    
    if success_rate >= 0.9:
        print("[PASS] Robustness: >90% success rate")
    else:
        print("[FAIL] Robustness: <90% success rate")
        all_passed = False
    
    if violations == 0:
        print("[PASS] Safety: No violations")
    else:
        print(f"[WARN] Safety: {violations} violations")
    
    if max_change <= 0.5:
        print("[PASS] Smoothness: Actions are smooth")
    else:
        print("[WARN] Smoothness: Large action changes detected")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: Policy is READY for real-world deployment")
    else:
        print("RESULT: Policy needs improvement before deployment")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    args = parser.parse_args()
    
    validate_policy(args.model_path, args.num_episodes)
```

---

# ملخص الخطة التنفيذية

| المرحلة | الأسبوع | المخرجات الرئيسية |
|---------|---------|-------------------|
| 1 | 1-2 | هيكل المشروع، الشبكة العصبية، نظام التكوين |
| 2 | 3-4 | L2F Wrapper، Replay Buffer، SAC |
| 3 | 5-6 | 1000 معلم مدرب |
| 4 | 7-8 | سياسة طالب موحدة |
| 5 | 9 | ONNX، TFLite |
| 6 | 10 | اختبارات، توثيق |
| **إضافي** | - | Trajectory Tracking، WebSocket UI، Safety، Android Integration |

---

# قائمة التحقق النهائية

## قبل البدء:
- [ ] GPU متاح (RTX 3090 موصى)
- [ ] Python 3.10+ مثبت
- [ ] PyTorch مع CUDA مثبت
- [ ] مساحة تخزين كافية (100 GB)

## بعد الانتهاء:
- [ ] جميع الاختبارات تمر
- [ ] النموذج يعمل في المحاكاة
- [ ] التصدير إلى ONNX ناجح
- [ ] التصدير إلى TFLite ناجح
- [ ] التوثيق كامل
- [ ] الكود نظيف ومنظم

## للنشر الواقعي (إضافي):
- [ ] Trajectory Tracking يعمل (50% hover + 50% Langevin)
- [ ] Observation/Action noise مُفعّل
- [ ] WebSocket UI يعمل للعرض ثلاثي الأبعاد
- [ ] اختبار الاستدلال بمعدل 250 Hz ناجح
- [ ] فحوصات السلامة مُفعّلة
- [ ] تحويل الإحداثيات (FLU) صحيح
- [ ] ترتيب المحركات مُتحقق منه
- [ ] تكامل Android كامل
- [ ] اختبار Sim-to-Real ناجح (>90% success rate)

---

**نهاية الخطة التنفيذية**

*تم إنشاء هذه الخطة كدليل تفصيلي لتنفيذ هجرة Raptor Foundation Policy من C++ إلى Python/PyTorch.*
*تم تحديث الخطة لتشمل جميع المكونات الضرورية للنشر على طائرة حقيقية.*
