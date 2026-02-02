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

