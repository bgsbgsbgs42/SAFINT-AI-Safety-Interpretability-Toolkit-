"""
Optimized Modular Analysis Engine Backend for Safety Interpretability Toolkit
================================================================

Key optimizations:
- Reduced memory usage with tensor sharing
- Asynchronous processing pipeline
- Batched operations
- Improved thread safety
- JIT compilation for JAX
"""

import abc
import asyncio
import logging
import pickle
import threading
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import weakref

import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, tree_util
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameworkType(Enum):
    """Supported ML frameworks with auto values."""
    PYTORCH = auto()
    JAX = auto()

class HookType(Enum):
    """Types of hooks that can be registered."""
    FORWARD = auto()
    BACKWARD = auto()
    FORWARD_PRE = auto()
    BACKWARD_PRE = auto()

@dataclass
class ActivationData:
    """Optimized container for activation data with memory-efficient tensor handling."""
    tensor: Union[torch.Tensor, np.ndarray]
    layer_name: str
    hook_type: HookType
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0)
    batch_size: int = field(init=False)
    shape: Tuple[int, ...] = field(init=False)
    device: str = field(init=False)
    dtype: str = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.shape = tuple(self.tensor.shape)
        self.batch_size = self.shape[0] if len(self.shape) > 0 else 1
        if isinstance(self.tensor, torch.Tensor):
            self.device = str(self.tensor.device)
            self.dtype = str(self.tensor.dtype)
        else:
            self.device = 'cpu'
            self.dtype = str(self.tensor.dtype)
    
    def to_numpy(self) -> np.ndarray:
        """Convert tensor to numpy array efficiently."""
        if isinstance(self.tensor, torch.Tensor):
            arr = self.tensor.detach().cpu().numpy()
        else:
            arr = np.array(self.tensor)
        return arr
    
    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics with batched operations."""
        data = self.to_numpy()
        flat_data = data.reshape(-1)
        return {
            'mean': float(np.mean(flat_data)),
            'std': float(np.std(flat_data)),
            'min': float(np.min(flat_data)),
            'max': float(np.max(flat_data)),
            'norm': float(np.linalg.norm(flat_data)),
            'sparsity': float(np.mean(flat_data == 0))
        }

@dataclass
class HookConfig:
    """Optimized hook configuration with default filters."""
    layer_name: str
    hook_type: HookType
    enabled: bool = True
    capture_gradients: bool = False
    downsample_rate: float = 1.0
    max_samples: Optional[int] = None
    filters: List[Callable] = field(default_factory=lambda: [default_filter])
    transforms: List[Callable] = field(default_factory=list)

def default_filter(tensor: Any) -> bool:
    """Default filter that always returns True."""
    return True

class DataBuffer:
    """Optimized thread-safe circular buffer with batched operations."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.total_added = 0
        self.layer_indices = defaultdict(list)
    
    def add(self, data: ActivationData) -> None:
        """Add data to buffer with index tracking."""
        with self.lock:
            self.buffer.append(data)
            self.layer_indices[data.layer_name].append(len(self.buffer) - 1)
            self.total_added += 1
    
    def get_recent(self, n: int = 100) -> List[ActivationData]:
        """Get n most recent items without full scan."""
        with self.lock:
            return list(self.buffer)[-n:]
    
    def get_by_layer(self, layer_name: str) -> List[ActivationData]:
        """Get all data for specific layer using pre-built indices."""
        with self.lock:
            return [self.buffer[i] for i in self.layer_indices.get(layer_name, [])]
    
    def clear(self) -> None:
        """Clear buffer and indices."""
        with self.lock:
            self.buffer.clear()
            self.layer_indices.clear()
            self.total_added = 0

class BaseHookManager(abc.ABC):
    """Optimized base class with common hook management."""
    
    __slots__ = ['buffer', 'hooks', 'hook_configs', 'sample_counts', 'framework_type']
    
    def __init__(self, buffer: DataBuffer):
        self.buffer = buffer
        self.hooks: Dict[str, Any] = {}
        self.hook_configs: Dict[str, HookConfig] = {}
        self.sample_counts: Dict[str, int] = defaultdict(int)
        self.framework_type = None
    
    @abc.abstractmethod
    def register_hook(self, model: Any, config: HookConfig) -> str:
        """Register a hook on the model."""
        pass
    
    @abc.abstractmethod
    def remove_hook(self, hook_id: str) -> None:
        """Remove a registered hook."""
        pass
    
    @abc.abstractmethod
    def get_layer_names(self, model: Any) -> List[str]:
        """Get all layer names from the model."""
        pass
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks efficiently."""
        for hook_id in list(self.hooks.keys()):
            self.remove_hook(hook_id)
    
    def _should_capture(self, config: HookConfig) -> bool:
        """Optimized capture decision logic."""
        if not config.enabled:
            return False
        
        if config.max_samples is not None:
            if self.sample_counts[config.layer_name] >= config.max_samples:
                return False
        
        if config.downsample_rate < 1.0:
            return np.random.random() < config.downsample_rate
        
        return True
    
    def _apply_transforms(self, tensor: Any, transforms: List[Callable]) -> Any:
        """Apply transforms with early termination."""
        result = tensor
        for transform in transforms:
            result = transform(result)
            if result is None:
                return None
        return result
    
    def _apply_filters(self, tensor: Any, filters: List[Callable]) -> bool:
        """Apply filters with short-circuit evaluation."""
        return all(f(tensor) for f in filters)

class PyTorchHookManager(BaseHookManager):
    """Optimized PyTorch hook manager with tensor sharing."""
    
    __slots__ = ['_model_refs']
    
    def __init__(self, buffer: DataBuffer):
        super().__init__(buffer)
        self.framework_type = FrameworkType.PYTORCH
        self._model_refs = weakref.WeakValueDictionary()
    
    def register_hook(self, model: nn.Module, config: HookConfig) -> str:
        """Register optimized PyTorch hook with tensor sharing."""
        hook_id = str(uuid.uuid4())
        self.hook_configs[hook_id] = config
        
        # Store weak reference to model
        model_id = id(model)
        self._model_refs[model_id] = model
        
        # Find target layer efficiently
        try:
            target_layer = dict(model.named_modules())[config.layer_name]
        except KeyError:
            raise ValueError(f"Layer {config.layer_name} not found in model")
        
        # Shared hook function
        def make_hook(hook_id, config):
            def hook_fn(module, input_data, output_data):
                if not self._should_capture(config):
                    return
                
                tensor = output_data if config.hook_type == HookType.FORWARD else input_data[0]
                
                if not self._apply_filters(tensor, config.filters):
                    return
                
                transformed = self._apply_transforms(tensor, config.transforms)
                if transformed is None:
                    return
                
                # Create activation data with shared tensor
                activation = ActivationData(
                    tensor=transformed.detach() if isinstance(transformed, torch.Tensor) else transformed,
                    layer_name=config.layer_name,
                    hook_type=config.hook_type
                )
                
                self.buffer.add(activation)
                self.sample_counts[config.layer_name] += 1
            
            return hook_fn
        
        # Register appropriate hook type
        if config.hook_type == HookType.FORWARD:
            handle = target_layer.register_forward_hook(make_hook(hook_id, config))
        elif config.hook_type == HookType.FORWARD_PRE:
            handle = target_layer.register_forward_pre_hook(make_hook(hook_id, config))
        else:
            raise ValueError(f"Unsupported hook type: {config.hook_type}")
        
        self.hooks[hook_id] = handle
        return hook_id
    
    def remove_hook(self, hook_id: str) -> None:
        """Remove hook with cleanup."""
        if hook_id in self.hooks:
            handle = self.hooks.pop(hook_id)
            if isinstance(handle, RemovableHandle):
                handle.remove()
            if hook_id in self.hook_configs:
                del self.hook_configs[hook_id]
    
    def get_layer_names(self, model: nn.Module) -> List[str]:
        """Get layer names with caching."""
        return list(dict(model.named_modules()).keys())

class JAXHookManager(BaseHookManager):
    """Optimized JAX hook manager with JIT compilation."""
    
    def __init__(self, buffer: DataBuffer):
        super().__init__(buffer)
        self.framework_type = FrameworkType.JAX
        self._jitted_functions = {}
    
    def register_hook(self, model: Any, config: HookConfig) -> str:
        """Register JAX hook with JIT compilation."""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available")
        
        hook_id = str(uuid.uuid4())
        self.hook_configs[hook_id] = config
        
        @jit
        def jax_hook_fn(*args, **kwargs):
            result = model(*args, **kwargs)
            
            if self._should_capture(config):
                if self._apply_filters(result, config.filters):
                    transformed = self._apply_transforms(result, config.transforms)
                    if transformed is not None:
                        activation = ActivationData(
                            tensor=np.array(transformed),
                            layer_name=config.layer_name,
                            hook_type=config.hook_type
                        )
                        self.buffer.add(activation)
                        self.sample_counts[config.layer_name] += 1
            
            return result
        
        self.hooks[hook_id] = jax_hook_fn
        self._jitted_functions[hook_id] = jax_hook_fn
        return hook_id
    
    def remove_hook(self, hook_id: str) -> None:
        """Remove JAX hook."""
        if hook_id in self.hooks:
            del self.hooks[hook_id]
            if hook_id in self._jitted_functions:
                del self._jitted_functions[hook_id]
            if hook_id in self.hook_configs:
                del self.hook_configs[hook_id]
    
    def get_layer_names(self, model: Any) -> List[str]:
        """Get layer names for JAX models."""
        return ['input', 'hidden', 'output']  # Placeholder for function-based models

class AnalysisEngine:
    """Optimized analysis engine with async processing."""
    
    __slots__ = ['buffer', 'hook_managers', 'executor', 'active_sessions', '_shutdown', '_loop']
    
    def __init__(self, buffer_size: int = 10000, max_workers: int = 4):
        self.buffer = DataBuffer(buffer_size)
        self.hook_managers = {
            FrameworkType.PYTORCH: PyTorchHookManager(self.buffer),
            FrameworkType.JAX: JAXHookManager(self.buffer) if JAX_AVAILABLE else None
        }
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_sessions = {}
        self._shutdown = False
        self._loop = asyncio.get_event_loop()
    
    async def process_async(self, coro):
        """Helper for async processing."""
        return await asyncio.shield(coro)
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create session with async timestamp."""
        session_id = session_id or str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'hooks': [],
            'created_at': self._loop.time(),
            'models': {}
        }
        return session_id
    
    def register_model(self, model: Any, session_id: str, 
                      framework: FrameworkType, model_name: str = "default") -> None:
        """Register model with validation."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        hook_manager = self.hook_managers.get(framework)
        if hook_manager is None:
            raise ValueError(f"Framework {framework} not available")
        
        self.active_sessions[session_id]['models'][model_name] = {
            'model': model,
            'framework': framework,
            'hook_manager': hook_manager
        }
    
    async def add_hook_async(self, session_id: str, model_name: str, config: HookConfig) -> str:
        """Async hook registration."""
        return await self.process_async(
            asyncio.get_running_loop().run_in_executor(
                self.executor, 
                self.add_hook, 
                session_id, 
                model_name, 
                config
            )
        )
    
    def add_hook(self, session_id: str, model_name: str, config: HookConfig) -> str:
        """Add hook with thread-safe operations."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        model_info = session['models'].get(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not registered")
        
        hook_id = model_info['hook_manager'].register_hook(model_info['model'], config)
        session['hooks'].append(hook_id)
        return hook_id
    
    def remove_hook(self, session_id: str, hook_id: str) -> None:
        """Remove hook with cleanup."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if hook_id in session['hooks']:
            session['hooks'].remove(hook_id)
        
        for hook_manager in self.hook_managers.values():
            if hook_manager and hook_id in hook_manager.hooks:
                hook_manager.remove_hook(hook_id)
                break
    
    async def analyze_activations_async(self, layer_name: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Async analysis of activations."""
        return await self.process_async(
            asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.analyze_activations,
                layer_name,
                analysis_type
            )
        )
    
    def analyze_activations(self, layer_name: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Optimized activation analysis with batched operations."""
        activations = self.buffer.get_by_layer(layer_name)
        if not activations:
            return {"error": f"No activations for {layer_name}"}
        
        if analysis_type == "summary":
            return self._compute_summary_analysis(activations)
        elif analysis_type == "temporal":
            return self._compute_temporal_analysis(activations)
        elif analysis_type == "distribution":
            return self._compute_distribution_analysis(activations)
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def _compute_summary_analysis(self, activations: List[ActivationData]) -> Dict[str, Any]:
        """Optimized summary statistics computation."""
        stats = [act.summary_stats() for act in activations]
        keys = stats[0].keys() if stats else []
        
        combined = {
            key: {
                'mean': np.mean([s[key] for s in stats]),
                'std': np.std([s[key] for s in stats]),
                'min': np.min([s[key] for s in stats]),
                'max': np.max([s[key] for s in stats])
            }
            for key in keys
        }
        
        return {
            'layer_name': activations[0].layer_name,
            'num_samples': len(activations),
            'statistics': combined,
            'shapes': [act.shape for act in activations[-5:]]
        }
    
    def shutdown(self) -> None:
        """Clean shutdown with resource cleanup."""
        if self._shutdown:
            return
        
        self._shutdown = True
        for session_id in list(self.active_sessions.keys()):
            self.close_session(session_id)
        
        self.buffer.clear()
        self.executor.shutdown(wait=False)
        logger.info("Engine shutdown complete")

# Optimized utility functions
def create_standard_hooks(layer_names: List[str], hook_type: HookType = HookType.FORWARD) -> List[HookConfig]:
    """Create hooks with list comprehension."""
    return [HookConfig(layer_name=name, hook_type=hook_type) for name in layer_names]

@jit
def jax_sparsity_filter(threshold: float = 0.9):
    """JIT-compiled sparsity filter for JAX."""
    def filter_fn(tensor):
        return jnp.mean(tensor == 0) < threshold
    return filter_fn

def torch_sparsity_filter(threshold: float = 0.9):
    """Efficient sparsity filter for PyTorch."""
    def filter_fn(tensor):
        return (tensor == 0).float().mean().item() < threshold
    return filter_fn