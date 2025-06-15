"""
Optimized Plugin System with Activation & Preference Integration
==============================================================

Enhanced plugin architecture that integrates with:
1. Activation data pipeline
2. Preference inconsistency detection
3. Safety analysis
"""

import asyncio
import importlib
import inspect
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Type
import numpy as np
import torch
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from previous modules
from activation_data_pipeline import ActivationBatch, StreamingConfig
from preference_inconsistency import PreferenceAnalyzer

class PluginType(Enum):
    """Optimized plugin types with integration support."""
    ACTIVATION_ANALYZER = "activation_analyzer"
    PREFERENCE_ANALYZER = "preference_analyzer"
    SAFETY_MONITOR = "safety_monitor"
    VISUALIZATION = "visualization"
    DATA_EXPORTER = "data_exporter"

class ExecutionMode(Enum):
    """Simplified execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"

@dataclass
class PluginMetadata:
    """Optimized plugin metadata."""
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict = field(default_factory=dict)
    priority: int = 50

@dataclass 
class PluginContext:
    """Enhanced context with activation and preference data."""
    session_id: str
    timestamp: float
    activation_data: Optional[torch.Tensor] = None
    preference_data: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    shared_state: Dict = field(default_factory=dict)

@dataclass
class PluginResult:
    """Standardized plugin result format."""
    success: bool
    data: Optional[Dict] = None
    metrics: Optional[Dict] = None
    alerts: Optional[List[Dict]] = None
    execution_time: float = 0.0

class BasePlugin:
    """Optimized base plugin class with integration support."""
    
    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.initialized = False
        self.stats = {
            'executions': 0,
            'successes': 0,
            'avg_time': 0.0
        }
    
    def initialize(self) -> bool:
        """Initialize plugin resources."""
        self.initialized = True
        return True
    
    def execute(self, context: PluginContext) -> PluginResult:
        """Execute plugin with timing and error handling."""
        start_time = time.time()
        result = PluginResult(success=False)
        
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            result = self._execute(context)
            result.success = True
            result.execution_time = time.time() - start_time
            
            # Update stats
            self.stats['executions'] += 1
            self.stats['successes'] += 1
            self.stats['avg_time'] = (
                (self.stats['avg_time'] * (self.stats['executions'] - 1) + 
                result.execution_time
            ) / self.stats['executions']
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            logger.error(f"Plugin {self.metadata.name} failed: {str(e)}")
            
        return result
    
    def _execute(self, context: PluginContext) -> PluginResult:
        """Plugin-specific implementation."""
        raise NotImplementedError()
    
    def cleanup(self):
        """Cleanup plugin resources."""
        self.initialized = False

class ActivationAnalyzerPlugin(BasePlugin):
    """For analyzing neural network activations."""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="activation_analyzer",
            version="1.0",
            plugin_type=PluginType.ACTIVATION_ANALYZER
        )
    
    def _execute(self, context: PluginContext) -> PluginResult:
        if context.activation_data is None:
            raise ValueError("No activation data provided")
        
        # Example analysis
        activations = context.activation_data.cpu().numpy()
        mean = float(np.mean(activations))
        std = float(np.std(activations))
        
        return PluginResult(
            success=True,
            metrics={
                'mean_activation': mean,
                'std_activation': std,
                'shape': list(activations.shape)
            }
        )

class PreferenceAnalyzerPlugin(BasePlugin):
    """Integrated preference inconsistency analyzer."""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="preference_analyzer",
            version="1.0",
            plugin_type=PluginType.PREFERENCE_ANALYZER
        )
        self.analyzer = PreferenceAnalyzer()
    
    def _execute(self, context: PluginContext) -> PluginResult:
        if context.preference_data is None:
            raise ValueError("No preference data provided")
        
        results = self.analyzer.detect_inconsistencies(context.preference_data)
        
        alerts = []
        if results.get('inconsistency_score', 0) > 0.7:  # Threshold
            alerts.append({
                'type': 'preference_inconsistency',
                'severity': 'high',
                'details': results
            })
        
        return PluginResult(
            success=True,
            data=results,
            alerts=alerts
        )

class SafetyMonitorPlugin(BasePlugin):
    """Integrated safety monitoring."""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="safety_monitor",
            version="1.0",
            plugin_type=PluginType.SAFETY_MONITOR
        )
        self.thresholds = {
            'activation_norm': 100.0,
            'sparsity': 0.9
        }
    
    def _execute(self, context: PluginContext) -> PluginResult:
        if context.activation_data is None:
            raise ValueError("No activation data provided")
        
        activations = context.activation_data.cpu().numpy()
        norm = np.linalg.norm(activations)
        sparsity = np.mean(activations == 0)
        
        alerts = []
        if norm > self.thresholds['activation_norm']:
            alerts.append({
                'type': 'high_activation_norm',
                'severity': 'medium',
                'value': float(norm),
                'threshold': self.thresholds['activation_norm']
            })
        
        if sparsity > self.thresholds['sparsity']:
            alerts.append({
                'type': 'high_sparsity',
                'severity': 'low',
                'value': float(sparsity),
                'threshold': self.thresholds['sparsity']
            })
        
        return PluginResult(
            success=True,
            metrics={
                'norm': float(norm),
                'sparsity': float(sparsity)
            },
            alerts=alerts
        )

class PluginManager:
    """Optimized plugin manager with integrated pipeline support."""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        # Register core plugins
        self.register(ActivationAnalyzerPlugin())
        self.register(PreferenceAnalyzerPlugin())
        self.register(SafetyMonitorPlugin())
    
    def register(self, plugin: BasePlugin) -> bool:
        """Register a plugin instance."""
        with self.lock:
            if plugin.metadata.name in self.plugins:
                logger.warning(f"Plugin {plugin.metadata.name} already registered")
                return False
            
            if plugin.initialize():
                self.plugins[plugin.metadata.name] = plugin
                logger.info(f"Registered plugin: {plugin.metadata.name}")
                return True
            return False
    
    def execute(self, plugin_name: str, context: PluginContext) -> PluginResult:
        """Execute a plugin by name."""
        plugin = self.plugins.get(plugin_name)
        if plugin is None:
            return PluginResult(success=False, data={"error": "Plugin not found"})
        
        if plugin.metadata.execution_mode == ExecutionMode.ASYNC:
            future = self.executor.submit(plugin.execute, context)
            return future.result()
        return plugin.execute(context)
    
    def analyze_activations(self, activations: torch.Tensor, 
                          metadata: Dict = None) -> Dict[str, PluginResult]:
        """Run activation analysis pipeline."""
        context = PluginContext(
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
            activation_data=activations,
            metadata=metadata or {}
        )
        
        results = {}
        for name, plugin in self.plugins.items():
            if plugin.metadata.plugin_type == PluginType.ACTIVATION_ANALYZER:
                results[name] = self.execute(name, context)
        
        return results
    
    def analyze_preferences(self, preference_data: Dict,
                          metadata: Dict = None) -> Dict[str, PluginResult]:
        """Run preference analysis pipeline."""
        context = PluginContext(
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
            preference_data=preference_data,
            metadata=metadata or {}
        )
        
        results = {}
        for name, plugin in self.plugins.items():
            if plugin.metadata.plugin_type == PluginType.PREFERENCE_ANALYZER:
                results[name] = self.execute(name, context)
        
        return results
    
    def run_safety_checks(self, activations: torch.Tensor,
                        preference_data: Optional[Dict] = None,
                        metadata: Dict = None) -> Dict[str, PluginResult]:
        """Run integrated safety checks."""
        context = PluginContext(
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
            activation_data=activations,
            preference_data=preference_data,
            metadata=metadata or {}
        )
        
        results = {}
        for name, plugin in self.plugins.items():
            if plugin.metadata.plugin_type == PluginType.SAFETY_MONITOR:
                results[name] = self.execute(name, context)
        
        return results

class IntegratedPipeline:
    """Combined pipeline for activation and preference analysis."""
    
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.stream_config = StreamingConfig()
        self.activation_buffer = []
        self.preference_buffer = []
    
    async def process_activation(self, activation: torch.Tensor,
                               metadata: Dict = None) -> Dict[str, PluginResult]:
        """Process activation data through all relevant plugins."""
        # Add to buffer for batch processing
        self.activation_buffer.append((activation, metadata or {}))
        
        if len(self.activation_buffer) >= self.stream_config.batch_size:
            return await self._process_batch()
        
        return {}
    
    async def process_preference(self, preference_data: Dict,
                               metadata: Dict = None) -> Dict[str, PluginResult]:
        """Process preference data through all relevant plugins."""
        self.preference_buffer.append((preference_data, metadata or {}))
        
        if len(self.preference_buffer) >= self.stream_config.batch_size:
            return await self._process_preference_batch()
        
        return {}
    
    async def _process_batch(self) -> Dict[str, PluginResult]:
        """Process a batch of activation data."""
        if not self.activation_buffer:
            return {}
        
        # Average the batch of activations for analysis
        activations = torch.stack([a for a, _ in self.activation_buffer])
        avg_activations = activations.mean(dim=0)
        
        metadata = {
            'batch_size': len(self.activation_buffer),
            **self.activation_buffer[0][1]  # First item's metadata
        }
        
        results = self.plugin_manager.analyze_activations(avg_activations, metadata)
        self.activation_buffer.clear()
        
        return results
    
    async def _process_preference_batch(self) -> Dict[str, PluginResult]:
        """Process a batch of preference data."""
        if not self.preference_buffer:
            return {}
        
        # Combine preference data
        combined_prefs = {
            'choices': [p['choice'] for p, _ in self.preference_buffer],
            'contexts': [p.get('context') for p, _ in self.preference_buffer],
            'timestamps': [p.get('timestamp') for p, _ in self.preference_buffer]
        }
        
        metadata = {
            'batch_size': len(self.preference_buffer),
            **self.preference_buffer[0][1]  # First item's metadata
        }
        
        results = self.plugin_manager.analyze_preferences(combined_prefs, metadata)
        self.preference_buffer.clear()
        
        return results
    
    async def run_safety_scan(self, activations: torch.Tensor,
                            preference_data: Optional[Dict] = None) -> Dict[str, PluginResult]:
        """Run integrated safety scan."""
        return self.plugin_manager.run_safety_checks(
            activations,
            preference_data
        )

# Example usage
async def demo_integrated_system():
    """Demonstrate the integrated system."""
    pipeline = IntegratedPipeline()
    
    # Simulate activation data
    activations = torch.randn(32, 768)  # Batch of 32, 768 features
    
    # Simulate preference data
    preferences = {
        'choice': 'option_a',
        'confidence': 0.8,
        'context': 'individual'
    }
    
    # Process activation data
    activation_results = await pipeline.process_activation(activations)
    print("Activation analysis results:")
    for name, result in activation_results.items():
        print(f"  {name}: {result.metrics}")
    
    # Process preference data
    preference_results = await pipeline.process_preference(preferences)
    print("\nPreference analysis results:")
    for name, result in preference_results.items():
        print(f"  {name}: {result.data}")
    
    # Run integrated safety check
    safety_results = await pipeline.run_safety_scan(activations, preferences)
    print("\nSafety check results:")
    for name, result in safety_results.items():
        if result.alerts:
            print(f"  {name} alerts: {len(result.alerts)}")
        else:
            print(f"  {name}: No alerts")

if __name__ == "__main__":
    asyncio.run(demo_integrated_system())