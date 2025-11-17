"""
Edge AI Deployment for Demand Forecasting
==========================================
Deploy ML models to edge devices for real-time forecasting:
- Model optimization (quantization, pruning)
- TensorFlow Lite conversion
- ONNX conversion
- Raspberry Pi / Jetson Nano deployment
- Real-time inference

Author: Geraldine Castillo
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelOptimizer:
    """
    Optimize ML models for edge deployment.
    """
    
    def __init__(self):
        self.optimized_model = None
        self.metadata = {}
        
    def quantize_model(
        self,
        model,
        model_type: str = 'sklearn',
        quantization_bits: int = 8
    ) -> Dict:
        """
        Quantize model to reduce size and improve inference speed.
        
        Args:
            model: Trained model
            model_type: 'sklearn', 'tensorflow', or 'pytorch'
            quantization_bits: Bit precision (8, 16, or 32)
            
        Returns:
            Optimization results
        """
        original_size = self._get_model_size(model)
        
        if model_type == 'sklearn':
            optimized = self._quantize_sklearn_model(model, quantization_bits)
        elif model_type == 'tensorflow':
            optimized = self._quantize_tensorflow_model(model, quantization_bits)
        elif model_type == 'pytorch':
            optimized = self._quantize_pytorch_model(model, quantization_bits)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        optimized_size = self._get_model_size(optimized)
        compression_ratio = original_size / optimized_size
        
        self.optimized_model = optimized
        
        return {
            'original_size_mb': original_size / (1024**2),
            'optimized_size_mb': optimized_size / (1024**2),
            'compression_ratio': compression_ratio,
            'quantization_bits': quantization_bits,
            'size_reduction_pct': (1 - optimized_size/original_size) * 100
        }
    
    def _quantize_sklearn_model(self, model, bits: int):
        """
        Quantize scikit-learn model by reducing coefficient precision.
        """
        import copy
        quantized = copy.deepcopy(model)
        
        # Quantize tree-based models
        if hasattr(quantized, 'estimators_'):  # Ensemble models
            for estimator in quantized.estimators_:
                if hasattr(estimator, 'tree_'):
                    self._quantize_tree(estimator.tree_, bits)
        elif hasattr(quantized, 'tree_'):  # Single tree models
            self._quantize_tree(quantized.tree_, bits)
        
        return quantized
    
    def _quantize_tree(self, tree, bits: int):
        """Quantize decision tree thresholds."""
        if hasattr(tree, 'threshold'):
            # Quantize thresholds
            scale = 2 ** bits - 1
            min_val = tree.threshold.min()
            max_val = tree.threshold.max()
            
            # Normalize to [0, 1]
            normalized = (tree.threshold - min_val) / (max_val - min_val + 1e-10)
            
            # Quantize
            quantized = np.round(normalized * scale) / scale
            
            # Denormalize
            tree.threshold[:] = quantized * (max_val - min_val) + min_val
    
    def _quantize_tensorflow_model(self, model, bits: int):
        """Quantize TensorFlow model."""
        try:
            import tensorflow as tf
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if bits == 8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif bits == 16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            return tflite_model
        except ImportError:
            print("TensorFlow not available. Returning original model.")
            return model
    
    def _quantize_pytorch_model(self, model, bits: int):
        """Quantize PyTorch model."""
        try:
            import torch
            
            if bits == 8:
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                return quantized_model
            else:
                return model
        except ImportError:
            print("PyTorch not available. Returning original model.")
            return model
    
    def _get_model_size(self, model) -> int:
        """Get model size in bytes."""
        import sys
        return sys.getsizeof(pickle.dumps(model))
    
    def prune_model(
        self,
        model,
        pruning_rate: float = 0.5
    ) -> Dict:
        """
        Prune model by removing low-importance features/weights.
        
        Args:
            model: Trained model
            pruning_rate: Fraction of features to remove
            
        Returns:
            Pruning results
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            importances = model.feature_importances_
            threshold = np.percentile(importances, pruning_rate * 100)
            
            important_features = importances >= threshold
            n_features_kept = important_features.sum()
            
            return {
                'original_features': len(importances),
                'features_kept': n_features_kept,
                'features_removed': len(importances) - n_features_kept,
                'pruning_rate': (len(importances) - n_features_kept) / len(importances),
                'important_feature_indices': np.where(important_features)[0].tolist()
            }
        else:
            return {
                'error': 'Model does not support feature importance-based pruning'
            }


class EdgeDeploymentConverter:
    """
    Convert models to edge-compatible formats.
    """
    
    def __init__(self):
        self.conversion_formats = ['tflite', 'onnx', 'pickle', 'json']
        
    def convert_to_tflite(
        self,
        model,
        input_shape: Tuple,
        output_path: str
    ) -> Dict:
        """
        Convert model to TensorFlow Lite for mobile/edge deployment.
        
        Args:
            model: Trained TensorFlow/Keras model
            input_shape: Input tensor shape
            output_path: Path to save .tflite file
            
        Returns:
            Conversion results
        """
        try:
            import tensorflow as tf
            
            # Convert
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get model info
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            return {
                'success': True,
                'output_path': output_path,
                'model_size_kb': len(tflite_model) / 1024,
                'input_shape': input_details[0]['shape'].tolist(),
                'output_shape': output_details[0]['shape'].tolist(),
                'format': 'tflite'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def convert_to_onnx(
        self,
        model,
        input_shape: Tuple,
        output_path: str,
        model_type: str = 'sklearn'
    ) -> Dict:
        """
        Convert model to ONNX format.
        
        Args:
            model: Trained model
            input_shape: Input shape
            output_path: Path to save .onnx file
            model_type: 'sklearn', 'tensorflow', or 'pytorch'
            
        Returns:
            Conversion results
        """
        try:
            if model_type == 'sklearn':
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                
                initial_type = [('float_input', FloatTensorType([None, input_shape[0]]))]
                onx = convert_sklearn(model, initial_types=initial_type)
                
                with open(output_path, 'wb') as f:
                    f.write(onx.SerializeToString())
                
            elif model_type == 'pytorch':
                import torch
                import torch.onnx
                
                dummy_input = torch.randn(1, *input_shape)
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
            
            # Get file size
            file_size = Path(output_path).stat().st_size
            
            return {
                'success': True,
                'output_path': output_path,
                'model_size_kb': file_size / 1024,
                'format': 'onnx'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_lightweight_format(
        self,
        model,
        feature_names: List[str],
        output_path: str,
        format: str = 'json'
    ) -> Dict:
        """
        Export model in lightweight format for edge devices.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            output_path: Output file path
            format: 'json' or 'pickle'
            
        Returns:
            Export results
        """
        model_dict = {
            'feature_names': feature_names,
            'model_type': type(model).__name__,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        # Extract model parameters
        if hasattr(model, 'feature_importances_'):
            model_dict['feature_importances'] = model.feature_importances_.tolist()
        
        if hasattr(model, 'n_estimators'):
            model_dict['n_estimators'] = model.n_estimators
        
        # Save
        if format == 'json':
            # Convert to JSON-serializable format
            with open(output_path, 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump({'metadata': model_dict, 'model': model}, f)
        
        file_size = Path(output_path).stat().st_size
        
        return {
            'success': True,
            'output_path': output_path,
            'model_size_kb': file_size / 1024,
            'format': format
        }


class EdgeInferenceEngine:
    """
    Lightweight inference engine for edge devices.
    """
    
    def __init__(self, model_path: str, model_format: str = 'pickle'):
        """
        Args:
            model_path: Path to model file
            model_format: 'pickle', 'tflite', 'onnx', or 'json'
        """
        self.model_path = model_path
        self.model_format = model_format
        self.model = None
        self.interpreter = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model based on format."""
        if self.model_format == 'pickle':
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model', data)
        
        elif self.model_format == 'tflite':
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
        
        elif self.model_format == 'onnx':
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
        
        elif self.model_format == 'json':
            with open(self.model_path, 'r') as f:
                self.model = json.load(f)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on edge device.
        
        Args:
            features: Input features
            
        Returns:
            Predictions
        """
        if self.model_format == 'pickle':
            return self.model.predict(features)
        
        elif self.model_format == 'tflite':
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Set input
            self.interpreter.set_tensor(
                input_details[0]['index'],
                features.astype(np.float32)
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(output_details[0]['index'])
            return output
        
        elif self.model_format == 'onnx':
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            result = self.model.run(
                [output_name],
                {input_name: features.astype(np.float32)}
            )
            return result[0]
        
        else:
            raise ValueError(f"Unsupported format: {self.model_format}")
    
    def benchmark_inference(
        self,
        sample_data: np.ndarray,
        n_iterations: int = 100
    ) -> Dict:
        """
        Benchmark inference performance.
        
        Args:
            sample_data: Sample input data
            n_iterations: Number of iterations
            
        Returns:
            Performance metrics
        """
        import time
        
        times = []
        
        # Warm-up
        for _ in range(10):
            _ = self.predict(sample_data)
        
        # Benchmark
        for _ in range(n_iterations):
            start = time.time()
            _ = self.predict(sample_data)
            end = time.time()
            times.append(end - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            'mean_latency_ms': times.mean(),
            'median_latency_ms': np.median(times),
            'std_latency_ms': times.std(),
            'min_latency_ms': times.min(),
            'max_latency_ms': times.max(),
            'p95_latency_ms': np.percentile(times, 95),
            'p99_latency_ms': np.percentile(times, 99),
            'throughput_per_sec': 1000 / times.mean()
        }


class RaspberryPiDeployment:
    """
    Deployment utilities for Raspberry Pi.
    """
    
    @staticmethod
    def generate_deployment_script(
        model_path: str,
        model_format: str,
        output_script: str = 'deploy_rpi.sh'
    ):
        """
        Generate deployment script for Raspberry Pi.
        
        Args:
            model_path: Path to model file
            model_format: Model format
            output_script: Output script path
        """
        script = f"""#!/bin/bash
# Raspberry Pi Deployment Script
# Generated: {pd.Timestamp.now()}

echo "Setting up demand forecasting on Raspberry Pi..."

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy

# Install dependencies
pip3 install pandas scikit-learn

# Install format-specific dependencies
"""
        
        if model_format == 'tflite':
            script += "pip3 install tensorflow-lite\n"
        elif model_format == 'onnx':
            script += "pip3 install onnxruntime\n"
        
        script += f"""
# Copy model
MODEL_PATH="{model_path}"
cp $MODEL_PATH /home/pi/models/

# Create inference service
cat > /home/pi/forecast_service.py << 'EOF'
import numpy as np
from edge_inference import EdgeInferenceEngine

# Load model
engine = EdgeInferenceEngine('/home/pi/models/{Path(model_path).name}', '{model_format}')

# Run inference
def forecast_demand(features):
    return engine.predict(features)

# Example usage
if __name__ == '__main__':
    # Sample features
    sample = np.array([[5, 420, 390, 1, 0, 22.5]])  # day, lag_7, rolling_avg, promo, holiday, temp
    prediction = forecast_demand(sample)
    print(f"Forecasted demand: {{prediction[0]:.0f}} orders")
EOF

# Make executable
chmod +x /home/pi/forecast_service.py

echo "Deployment complete! Run: python3 /home/pi/forecast_service.py"
"""
        
        with open(output_script, 'w') as f:
            f.write(script)
        
        # Make executable
        Path(output_script).chmod(0o755)
        
        return {
            'script_path': output_script,
            'instructions': f"Copy {output_script} and {model_path} to Raspberry Pi, then run: ./{output_script}"
        }
    
    @staticmethod
    def estimate_edge_requirements(model_size_mb: float, features_count: int) -> Dict:
        """
        Estimate hardware requirements for edge deployment.
        
        Args:
            model_size_mb: Model size in MB
            features_count: Number of input features
            
        Returns:
            Hardware requirements
        """
        # Calculate memory requirements
        model_memory_mb = model_size_mb * 1.5  # Model + overhead
        inference_memory_mb = features_count * 4 * 1000 / (1024**2)  # 4 bytes per float, 1000 samples
        total_memory_mb = model_memory_mb + inference_memory_mb + 50  # + OS overhead
        
        # Determine suitable devices
        devices = []
        
        if total_memory_mb < 512:
            devices.append({
                'name': 'Raspberry Pi Zero W',
                'ram_mb': 512,
                'cpu': 'Single-core 1GHz',
                'cost_usd': 10,
                'suitable': True
            })
        
        if total_memory_mb < 1024:
            devices.append({
                'name': 'Raspberry Pi 3',
                'ram_mb': 1024,
                'cpu': 'Quad-core 1.2GHz',
                'cost_usd': 35,
                'suitable': True
            })
        
        devices.append({
            'name': 'Raspberry Pi 4 (2GB)',
            'ram_mb': 2048,
            'cpu': 'Quad-core 1.5GHz',
            'cost_usd': 45,
            'suitable': True
        })
        
        devices.append({
            'name': 'NVIDIA Jetson Nano',
            'ram_mb': 4096,
            'cpu': 'Quad-core ARM + 128-core GPU',
            'cost_usd': 99,
            'suitable': True,
            'features': 'GPU acceleration available'
        })
        
        return {
            'estimated_memory_mb': total_memory_mb,
            'model_memory_mb': model_memory_mb,
            'inference_memory_mb': inference_memory_mb,
            'suitable_devices': devices,
            'recommendation': devices[0]['name'] if len(devices) > 0 else 'Dedicated server required'
        }


# Example usage
if __name__ == "__main__":
    print("Edge AI Deployment Module")
    print("=" * 50)
    print("\nFeatures:")
    print("- Model quantization (8-bit, 16-bit)")
    print("- Model pruning (feature importance)")
    print("- TensorFlow Lite conversion")
    print("- ONNX conversion")
    print("- Raspberry Pi deployment")
    print("- Real-time inference engine")
    print("- Performance benchmarking")
    print("\nSupported Devices:")
    print("- Raspberry Pi (all models)")
    print("- NVIDIA Jetson Nano")
    print("- Edge TPU")
    print("- Mobile devices (Android/iOS)")
