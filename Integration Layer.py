"""
Optimized Universal Integration Layer for ML Frameworks
=====================================================

A streamlined integration system with enhanced compatibility and performance.
"""

import abc
import logging
import os
import time
import uuid
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Framework imports with lazy loading
PYTORCH_AVAILABLE = TENSORFLOW_AVAILABLE = JAX_AVAILABLE = TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Framework(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    UNKNOWN = "unknown"

class ModelArchitecture(Enum):
    TRANSFORMER = "transformer"
    CNN = "cnn"
    MLP = "mlp"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    RESNET = "resnet"
    VIT = "vision_transformer"
    CUSTOM = "custom"

class ExportFormat(Enum):
    PNG = "png"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"

@dataclass
class ModelInfo:
    name: str
    framework: Framework
    architecture: ModelArchitecture
    layers: List[str]
    parameters: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationConfig:
    framework: Framework
    auto_detect_architecture: bool = True
    export_format: ExportFormat = ExportFormat.PNG
    export_dpi: int = 300
    figure_style: str = "publication"

class BaseModelAdapter(abc.ABC):
    """Base class for framework-specific model adapters with common optimizations."""
    
    def __init__(self, model: Any, config: IntegrationConfig):
        self.model = model
        self.config = config
        self.hooks: Dict[str, Any] = {}
        self.model_info: Optional[ModelInfo] = None
        self._setup_model()
        
    def _setup_model(self):
        """Framework-specific model setup."""
        pass
        
    @abc.abstractmethod
    def get_model_info(self) -> ModelInfo:
        pass
    
    @abc.abstractmethod
    def get_layer_names(self) -> List[str]:
        pass
    
    @abc.abstractmethod
    def add_activation_hook(self, layer_name: str, hook_fn: Callable) -> str:
        pass
    
    @abc.abstractmethod
    def remove_hook(self, hook_id: str) -> bool:
        pass
    
    @abc.abstractmethod
    def forward_pass(self, inputs: Any) -> Any:
        pass
    
    @abc.abstractmethod
    def get_layer_output(self, layer_name: str, inputs: Any) -> Any:
        pass

class PyTorchAdapter(BaseModelAdapter):
    """Optimized PyTorch adapter with reduced overhead."""
    
    def _setup_model(self):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        self.model.eval()
        
    def get_model_info(self) -> ModelInfo:
        if self.model_info:
            return self.model_info
        
        total_params = sum(p.numel() for p in self.model.parameters())
        layers = self.get_layer_names()
        
        self.model_info = ModelInfo(
            name=self.model.__class__.__name__,
            framework=Framework.PYTORCH,
            architecture=self._detect_architecture(),
            layers=layers,
            parameters=total_params,
            input_shape=self._get_sample_input_shape(),
            output_shape=self._get_sample_output_shape(),
            metadata={
                'num_layers': len(layers),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        )
        return self.model_info
    
    def get_layer_names(self) -> List[str]:
        return [name for name, _ in self.model.named_modules() if name]
    
    def add_activation_hook(self, layer_name: str, hook_fn: Callable) -> str:
        for name, module in self.model.named_modules():
            if name == layer_name:
                def wrapper_hook(_, __, output):
                    hook_fn(layer_name, output, {
                        'output_shape': output.shape,
                        'module_type': type(module).__name__
                    })
                handle = module.register_forward_hook(wrapper_hook)
                hook_id = str(uuid.uuid4())
                self.hooks[hook_id] = handle
                return hook_id
        raise ValueError(f"Layer {layer_name} not found")
    
    def remove_hook(self, hook_id: str) -> bool:
        if hook_id in self.hooks:
            self.hooks[hook_id].remove()
            del self.hooks[hook_id]
            return True
        return False
    
    def forward_pass(self, inputs: Any) -> Any:
        with torch.no_grad():
            return self.model(inputs)
    
    def get_layer_output(self, layer_name: str, inputs: Any) -> Any:
        activations = {}
        def capture_hook(_, __, output):
            activations[layer_name] = output
        
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(capture_hook)
                try:
                    with torch.no_grad():
                        self.model(inputs)
                    return activations.get(layer_name)
                finally:
                    handle.remove()
        raise ValueError(f"Layer {layer_name} not found")
    
    def _detect_architecture(self) -> ModelArchitecture:
        model_name = self.model.__class__.__name__.lower()
        layer_types = [type(module).__name__.lower() for _, module in self.model.named_modules()]
        
        if any(n in model_name for n in ['transformer', 'bert', 'gpt']):
            return ModelArchitecture.TRANSFORMER
        elif any(n in model_name for n in ['resnet', 'efficientnet']):
            return ModelArchitecture.CNN
        elif any(n in model_name for n in ['lstm']):
            return ModelArchitecture.LSTM
        elif any(n in model_name for n in ['gru']):
            return ModelArchitecture.GRU
        elif any(n in model_name for n in ['rnn']):
            return ModelArchitecture.RNN
        elif any('conv' in lt for lt in layer_types):
            return ModelArchitecture.CNN
        elif any('attention' in lt or 'multihead' in lt for lt in layer_types):
            return ModelArchitecture.TRANSFORMER
        elif any('linear' in lt for lt in layer_types):
            return ModelArchitecture.MLP
        return ModelArchitecture.CUSTOM
    
    def _get_sample_input_shape(self) -> Tuple[int, ...]:
        sample = self._create_sample_input()
        return sample.shape
    
    def _get_sample_output_shape(self) -> Tuple[int, ...]:
        sample = self._create_sample_input()
        with torch.no_grad():
            output = self.model(sample)
        return output.shape if isinstance(output, torch.Tensor) else output[0].shape
    
    def _create_sample_input(self) -> torch.Tensor:
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            return torch.randint(0, 1000, (1, 10))
        for shape in [(1, 3, 224, 224), (1, 784), (1, 10)]:
            try:
                sample = torch.randn(shape)
                with torch.no_grad():
                    self.model(sample)
                return sample
            except:
                continue
        return torch.randn(1, 10)

class TensorFlowAdapter(BaseModelAdapter):
    """Optimized TensorFlow/Keras adapter."""
    
    def get_model_info(self) -> ModelInfo:
        if self.model_info:
            return self.model_info
        
        self.model_info = ModelInfo(
            name=self.model.name,
            framework=Framework.TENSORFLOW,
            architecture=self._detect_architecture(),
            layers=self.get_layer_names(),
            parameters=self.model.count_params(),
            input_shape=self.model.input_shape[1:] if self.model.input_shape else (None,),
            output_shape=self.model.output_shape[1:] if self.model.output_shape else (None,),
            metadata={
                'num_layers': len(self.model.layers),
                'trainable_params': sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)
            }
        )
        return self.model_info
    
    def get_layer_names(self) -> List[str]:
        return [layer.name for layer in self.model.layers]
    
    def add_activation_hook(self, layer_name: str, hook_fn: Callable) -> str:
        for layer in self.model.layers:
            if layer.name == layer_name:
                hook_id = str(uuid.uuid4())
                self.hooks[hook_id] = {
                    'layer_name': layer_name,
                    'hook_fn': hook_fn,
                    'model': tf.keras.Model(inputs=self.model.input, outputs=layer.output)
                }
                return hook_id
        raise ValueError(f"Layer {layer_name} not found")
    
    def remove_hook(self, hook_id: str) -> bool:
        if hook_id in self.hooks:
            del self.hooks[hook_id]
            return True
        return False
    
    def forward_pass(self, inputs: Any) -> Any:
        return self.model(inputs, training=False)
    
    def get_layer_output(self, layer_name: str, inputs: Any) -> Any:
        for layer in self.model.layers:
            if layer.name == layer_name:
                return tf.keras.Model(inputs=self.model.input, outputs=layer.output)(inputs, training=False)
        raise ValueError(f"Layer {layer_name} not found")
    
    def _detect_architecture(self) -> ModelArchitecture:
        layer_types = [type(layer).__name__.lower() for layer in self.model.layers]
        
        if any('conv' in lt for lt in layer_types):
            return ModelArchitecture.CNN
        elif any('lstm' in lt for lt in layer_types):
            return ModelArchitecture.LSTM
        elif any('gru' in lt for lt in layer_types):
            return ModelArchitecture.GRU
        elif any('rnn' in lt for lt in layer_types):
            return ModelArchitecture.RNN
        elif any('attention' in lt for lt in layer_types):
            return ModelArchitecture.TRANSFORMER
        return ModelArchitecture.CUSTOM

class ModelRegistry:
    """Optimized model registry with caching."""
    
    def __init__(self):
        self.adapters: Dict[Framework, Type[BaseModelAdapter]] = {
            Framework.PYTORCH: PyTorchAdapter,
            Framework.TENSORFLOW: TensorFlowAdapter
        }
        self.registered_models: Dict[str, BaseModelAdapter] = {}
        
    def register_model(self, model_id: str, model: Any, config: Optional[IntegrationConfig] = None) -> BaseModelAdapter:
        framework = self._detect_framework(model)
        
        if config is None:
            config = IntegrationConfig(framework=framework)
        
        adapter_class = self.adapters.get(framework)
        if not adapter_class:
            raise ValueError(f"Unsupported framework: {framework}")
        
        adapter = adapter_class(model, config)
        self.registered_models[model_id] = adapter
        return adapter
    
    def get_model(self, model_id: str) -> Optional[BaseModelAdapter]:
        return self.registered_models.get(model_id)
    
    def _detect_framework(self, model: Any) -> Framework:
        if PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            return Framework.PYTORCH
        elif TENSORFLOW_AVAILABLE and isinstance(model, (tf.keras.Model, tf.keras.layers.Layer)):
            return Framework.TENSORFLOW
        return Framework.UNKNOWN

class ExportManager:
    """Optimized export manager with reduced dependencies."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        if self.config.figure_style == "publication":
            plt.style.use('seaborn-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'figure.dpi': self.config.export_dpi,
                'savefig.dpi': self.config.export_dpi
            })
    
    def export_figure(self, fig: plt.Figure, filename: str) -> str:
        output_path = Path(filename).with_suffix(f'.{self.config.export_format.value}')
        fig.savefig(output_path, bbox_inches='tight', dpi=self.config.export_dpi)
        plt.close(fig)
        return str(output_path)
    
    def export_data(self, data: Dict, filename: str) -> str:
        output_path = Path(filename).with_suffix('.json')
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return str(output_path)

class WorkflowIntegrator:
    """Core integration workflow with optimized operations."""
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.export_manager = None
        
    def setup_experiment(self, config: IntegrationConfig) -> None:
        self.export_manager = ExportManager(config)
    
    def integrate_model(self, model: Any, model_id: str = None, config: Optional[IntegrationConfig] = None) -> BaseModelAdapter:
        model_id = model_id or f"model_{len(self.model_registry.registered_models)}"
        return self.model_registry.register_model(model_id, model, config)
    
    def run_analysis(self, model_id: str, inputs: Any, layer_names: List[str] = None) -> Dict:
        adapter = self.model_registry.get_model(model_id)
        if not adapter:
            raise ValueError(f"Model {model_id} not found")
        
        results = {
            'model_info': adapter.get_model_info(),
            'activations': {},
            'analysis': {}
        }
        
        layer_names = layer_names or adapter.get_layer_names()[:10]
        
        for layer_name in layer_names:
            try:
                activation = adapter.get_layer_output(layer_name, inputs)
                activation = activation.numpy() if hasattr(activation, 'numpy') else np.array(activation)
                
                results['activations'][layer_name] = activation
                results['analysis'][layer_name] = {
                    'shape': activation.shape,
                    'mean': float(np.mean(activation)),
                    'std': float(np.std(activation))
                }
            except Exception as e:
                logger.warning(f"Layer {layer_name} analysis failed: {str(e)}")
        
        return results
    
    def export_results(self, results: Dict, base_filename: str) -> List[str]:
        if not self.export_manager:
            return []
        
        exports = []
        
        # Export analysis data
        data_path = self.export_manager.export_data(results, f"{base_filename}_data")
        exports.append(data_path)
        
        # Create and export visualizations
        if results.get('activations'):
            fig = self._create_activation_plot(results['activations'])
            fig_path = self.export_manager.export_figure(fig, f"{base_filename}_activations")
            exports.append(fig_path)
        
        return exports
    
    def _create_activation_plot(self, activations: Dict[str, np.ndarray]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        layer_names = list(activations.keys())
        means = [np.mean(act) for act in activations.values()]
        stds = [np.std(act) for act in activations.values()]
        
        ax.errorbar(range(len(layer_names)), means, yerr=stds, fmt='o-', capsize=5)
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_title('Layer Activation Analysis')
        ax.set_ylabel('Activation Value')
        ax.grid(True, alpha=0.3)
        
        return fig

def demo_optimized_integration():
    """Optimized demonstration of the integration layer."""
    integrator = WorkflowIntegrator()
    config = IntegrationConfig(
        framework=Framework.PYTORCH,
        export_format=ExportFormat.PNG
    )
    integrator.setup_experiment(config)
    
    if PYTORCH_AVAILABLE:
        # Simple test model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        
        # Integrate and analyze
        adapter = integrator.integrate_model(model, "test_model")
        test_input = torch.randn(1, 10)
        results = integrator.run_analysis("test_model", test_input)
        
        # Export results
        exports = integrator.export_results(results, "test_analysis")
        print(f"Analysis complete. Exports: {exports}")

if __name__ == "__main__":
    demo_optimized_integration()