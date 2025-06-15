# SAFINT: AI Safety Interpretability Toolkit

A comprehensive toolkit for neural network interpretability and safety analysis, designed specifically for AI safety researchers and practitioners.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Licence](#licence)

## 🔍 Overview

SAFINT provides a unified platform for analysing neural network safety properties through:

- **Real-time activation monitoring** across PyTorch, TensorFlow, and JAX models
- **Attention pattern analysis** with safety scoring for transformer architectures  
- **Preference inconsistency detection** to identify potential deceptive behaviours
- **Extensible plugin system** for custom safety analysis methods
- **Publication-ready visualisations** with multiple export formats

The toolkit is designed to integrate seamlessly into existing research workflows whilst providing powerful safety analysis capabilities.

## ✨ Features

### Core Analysis Engine
- **Multi-framework support**: PyTorch, TensorFlow, JAX compatibility
- **Real-time processing**: Streaming activation capture with configurable buffering
- **Memory optimisation**: Efficient handling of large activation datasets
- **Thread-safe operations**: Concurrent analysis with automatic resource management

### Safety Analysis
- **Anomaly detection**: Statistical outlier identification in activation patterns
- **Deception monitoring**: Behavioural consistency analysis
- **Preference analysis**: Detection of inconsistent preference expressions
- **Reward hacking detection**: Identification of specification gaming behaviours

### Visualisation & Export
- **Interactive dashboards**: Real-time monitoring with Streamlit interface
- **Publication-ready figures**: High-DPI exports in PNG, PDF, SVG, EPS formats
- **Attention visualisations**: Head-by-head analysis with safety scoring
- **Custom plot generation**: Extensible visualisation system

### Integration Layer
- **Model auto-detection**: Automatic architecture identification
- **Hook management**: Framework-agnostic activation capture
- **Workflow integration**: Easy integration into existing research pipelines
- **Export capabilities**: Multiple format support for research dissemination

## 🚀 Installation

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Core Dependencies

```bash
# Clone the repository
git clone https://github.com/your-org/safint.git
cd safint

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### Framework-Specific Dependencies

Install dependencies for your preferred ML framework:

```bash
# For PyTorch support
pip install torch torchvision torchaudio

# For TensorFlow support  
pip install tensorflow

# For JAX support
pip install jax jaxlib

# For Transformers support
pip install transformers
```

### Optional Dependencies

```bash
# For enhanced visualisation
pip install plotly dash

# For advanced data storage
pip install h5py zarr

# For performance optimisation
pip install lz4 psutil

# For preference analysis
pip install scikit-learn pandas
```

### Requirements File

Create a `requirements.txt` file:

```txt
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
streamlit>=1.25.0
plotly>=5.15.0
pandas>=1.5.0
pyyaml>=6.0
packaging>=21.0
asyncio-compat>=0.1.0
```

## 🏃 Quick Start

### 1. Basic Setup

```python
from safint import WorkflowIntegrator, IntegrationConfig, Framework

# Initialise the integrator
integrator = WorkflowIntegrator()

# Configure for your framework
config = IntegrationConfig(
    framework=Framework.PYTORCH,
    export_format="png",
    figure_style="publication"
)
integrator.setup_experiment(config)
```

### 2. Model Integration

```python
import torch
import torch.nn as nn

# Define your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Integrate with SAFINT
adapter = integrator.integrate_model(model, "my_model")
```

### 3. Run Analysis

```python
# Create test input
test_input = torch.randn(32, 784)

# Run comprehensive analysis
results = integrator.run_analysis_pipeline(
    "my_model", 
    test_input,
    export_results=True
)

# View results
print(f"Analysed {len(results['activations'])} layers")
print(f"Generated {len(results['exports'])} export files")
```

### 4. Launch Interactive Dashboard

```bash
# Start the Streamlit interface
streamlit run Frontend.py
```

Navigate to `http://localhost:8501` in your browser to access the interactive dashboard.

## 🧩 Core Components

### Backend Analysis Engine (`Backend.py`)

The core processing engine that handles:
- Activation capture through framework-specific hooks
- Thread-safe data buffering and processing
- Memory management and cleanup
- Asynchronous analysis pipeline

```python
from modular_analysis_engine import AnalysisEngine, HookConfig, HookType

# Create analysis engine
engine = AnalysisEngine(buffer_size=10000, max_workers=4)

# Create session
session_id = engine.create_session()

# Register model and add hooks
engine.register_model(model, session_id, Framework.PYTORCH)
hook_config = HookConfig(layer_name="linear1", hook_type=HookType.FORWARD)
engine.add_hook(session_id, "default", hook_config)
```

### Data Pipeline (`Data Pipeline.py`)

High-performance streaming pipeline for large datasets:
- Configurable compression and storage backends
- Real-time processing with backpressure handling
- Integration with preference analysis system
- Automatic cleanup and retention policies

```python
from activation_data_pipeline import StreamProcessor, StreamingConfig

# Configure pipeline
config = StreamingConfig(
    buffer_size=10000,
    batch_size=32,
    storage_backend='hdf5',
    enable_preference_analysis=True
)

# Start processor
processor = StreamProcessor(config)
processor.start()

# Submit data
processor.submit(
    session_id="experiment_1",
    layer_name="attention_layer",
    tensor=activation_tensor,
    metadata={"step": 1}
)
```

### Frontend Interface (`Frontend.py`)

Interactive Streamlit dashboard providing:
- Real-time activation monitoring
- Safety metric visualisation
- Alert system for concerning patterns
- Export functionality for research outputs

```bash
# Launch with custom configuration
streamlit run Frontend.py --server.port 8502 --server.headless true
```

### Integration Layer (`Integration Layer.py`)

Universal compatibility layer supporting:
- Automatic framework detection
- Model architecture analysis
- Publication-ready export management
- Research workflow integration

```python
from integration_layer import ModelRegistry, ExportManager

# Register models from different frameworks
registry = ModelRegistry()
pytorch_adapter = registry.register_model("pytorch_model", pytorch_model)
tf_adapter = registry.register_model("tf_model", tensorflow_model)

# Export analysis results
export_manager = ExportManager(config)
export_path = export_manager.export_figure(figure, "analysis_results")
```

### Plugin System (`Plug In.py`)

Extensible architecture for custom analysis methods:
- Plugin discovery and registration
- Execution mode management (sync/async/streaming)
- Integration with activation and preference pipelines
- Template generation for new plugins

```python
from plugin_system import PluginManager, BasePlugin

# Create custom plugin
class MyAnalyzer(BasePlugin):
    def _execute(self, context):
        # Your analysis logic here
        return PluginResult(success=True, metrics={"score": 0.85})

# Register and use
manager = PluginManager()
manager.register(MyAnalyzer())
results = manager.run_safety_checks(activations)
```

## 📚 Usage Examples

### Example 1: Basic Model Analysis

```python
import torch
import torch.nn as nn
from safint import WorkflowIntegrator, IntegrationConfig, Framework

# Setup
integrator = WorkflowIntegrator()
config = IntegrationConfig(framework=Framework.PYTORCH)
integrator.setup_experiment(config)

# Create model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Analyse
adapter = integrator.integrate_model(model, "test_model")
test_input = torch.randn(1, 10)
results = integrator.run_analysis("test_model", test_input)

print(f"Model has {results['model_info'].parameters:,} parameters")
print(f"Analysed {len(results['activations'])} layers")
```

### Example 2: Real-time Safety Monitoring

```python
from safint.frontend import SAFINTFrontend
import streamlit as st

# Initialise frontend
frontend = SAFINTFrontend()

# Register your model
frontend.register_model(your_model, Framework.PYTORCH, "production_model")

# Add monitoring hooks
layer_names = ["attention", "feedforward", "output"]
frontend.add_hooks(layer_names, "production_model")

# The dashboard will now show real-time analysis
# Access via streamlit run script
```

### Example 3: Custom Plugin Development

```python
from safint.plugins import BasePlugin, PluginMetadata, PluginType

class CustomSafetyAnalyzer(BasePlugin):
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="custom_analyzer",
            version="1.0.0",
            plugin_type=PluginType.SAFETY_ANALYZER,
            description="Custom safety analysis method"
        )
    
    def _execute(self, context):
        # Implement your analysis
        activations = context.activation_data
        safety_score = your_analysis_function(activations)
        
        return PluginResult(
            success=True,
            metrics={"safety_score": safety_score},
            alerts=[] if safety_score > 0.7 else [{"type": "low_safety"}]
        )

# Register and use
from safint import PluginManager
manager = PluginManager()
manager.register(CustomSafetyAnalyzer())
```

### Example 4: Batch Processing Pipeline

```python
import asyncio
from safint.pipeline import IntegratedPipeline

async def process_dataset(dataset):
    pipeline = IntegratedPipeline()
    
    results = []
    for batch in dataset:
        # Process activations
        activation_results = await pipeline.process_activation(
            batch.activations,
            metadata={"batch_id": batch.id}
        )
        
        # Process preferences if available
        if hasattr(batch, 'preferences'):
            preference_results = await pipeline.process_preference(
                batch.preferences
            )
        
        # Run safety checks
        safety_results = await pipeline.run_safety_scan(
            batch.activations,
            batch.preferences if hasattr(batch, 'preferences') else None
        )
        
        results.append({
            'activations': activation_results,
            'preferences': preference_results,
            'safety': safety_results
        })
    
    return results

# Usage
results = asyncio.run(process_dataset(your_dataset))
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# Storage configuration
SAFINT_STORAGE_BACKEND=hdf5
SAFINT_DATA_PATH=./data/activations
SAFINT_EXPORT_PATH=./exports

# Performance settings
SAFINT_MAX_WORKERS=4
SAFINT_BUFFER_SIZE=10000
SAFINT_MAX_MEMORY_GB=8.0

# Visualization settings
SAFINT_EXPORT_DPI=300
SAFINT_FIGURE_STYLE=publication
SAFINT_COLOR_PALETTE=viridis

# Safety thresholds
SAFINT_ANOMALY_THRESHOLD=0.7
SAFINT_SAFETY_THRESHOLD=0.5
```

### Configuration Files

Create `config.yaml` for advanced settings:

```yaml
analysis:
  frameworks:
    - pytorch
    - tensorflow
  architectures:
    - transformer
    - cnn
    - mlp
  
streaming:
  buffer_size: 10000
  batch_size: 32
  compression: lz4
  retention_hours: 24

safety:
  anomaly_threshold: 0.7
  deception_threshold: 0.6
  preference_threshold: 0.5
  
export:
  formats: [png, pdf, svg]
  dpi: 300
  style: publication

plugins:
  auto_discover: true
  plugin_paths:
    - ./plugins
    - ./custom_analyzers
```

### Dashboard Configuration

Streamlit configuration in `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

## 🔧 Troubleshooting

### Common Issues

#### ImportError: No module named 'torch'

**Solution**: Install PyTorch for your system:
```bash
# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA version (check CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues with Large Models

**Solution**: Adjust memory settings:
```python
config = IntegrationConfig(
    max_memory_gb=16.0,  # Increase memory limit
    buffer_size=5000     # Reduce buffer size
)
```

#### Streamlit Dashboard Not Loading

**Solution**: Check port availability and permissions:
```bash
# Try different port
streamlit run Frontend.py --server.port 8502

# Check if port is in use
netstat -tulpn | grep :8501
```

#### Plugin Loading Failures

**Solution**: Verify plugin structure:
```
your_plugin/
├── plugin.yaml
├── __init__.py
└── analyzer.py
```

#### Framework Detection Issues

**Solution**: Explicitly specify framework:
```python
config = IntegrationConfig(
    framework=Framework.PYTORCH,  # Explicit specification
    auto_detect_architecture=False
)
```

### Performance Optimisation

#### For Large Models
```python
# Use streaming configuration
config = StreamingConfig(
    buffer_size=50000,
    batch_size=64,
    compression='lz4',
    num_workers=8
)
```

#### For Real-time Analysis
```python
# Optimise for latency
config = IntegrationConfig(
    execution_mode=ExecutionMode.ASYNC,
    max_execution_time=1.0
)
```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export SAFINT_LOG_LEVEL=DEBUG
```

## 🤝 Contributing

We welcome contributions to SAFINT! Please follow these guidelines:

### Development Setup

```bash
# Clone development version
git clone https://github.com/your-org/safint.git
cd safint

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=safint --cov-report=html

# Run specific test categories
python -m pytest tests/test_backend.py -v
python -m pytest tests/test_frontend.py -v
python -m pytest tests/test_plugins.py -v
```

### Code Style

We follow PEP 8 with some modifications:
- Line length: 88 characters
- Use type hints
- Use British English in documentation
- Include docstrings for all public methods

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-analyzer`
3. Make your changes with tests
4. Run the test suite: `python -m pytest`
5. Submit a pull request with clear description

### Plugin Development

Create new plugins using our template:
```python
from safint.plugins import PluginTemplate

# Generate template
template = PluginTemplate.generate_safety_analyzer_template(
    plugin_name="my_analyzer",
    author="Your Name"
)

# Save to file
with open("my_analyzer.py", "w") as f:
    f.write(template)
```

## 📄 Licence

This project is licensed under the MIT Licence. See the [LICENCE](LICENCE) file for details.

## 🆘 Support

### Documentation
- [API Reference](docs/api.md)
- [Plugin Development Guide](docs/plugins.md)
- [Examples Repository](examples/)


---

**Note**: This toolkit is designed for research purposes. Please ensure appropriate validation before using in production safety-critical systems.

For the latest updates and documentation, visit our [GitHub repository](https://github.com/your-org/safint).
