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

