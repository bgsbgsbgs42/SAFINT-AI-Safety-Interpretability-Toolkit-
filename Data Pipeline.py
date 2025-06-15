"""
Optimized Activation Data Pipeline with Preference Integration
============================================================

Enhanced version that integrates with the preference inconsistency detection system
while maintaining high performance for neural network activation data processing.
"""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable, AsyncIterator
from pathlib import Path
import time
import json
import lz4.frame
import uuid
from collections import deque, defaultdict
import logging
import numpy as np
import torch
import h5py
import zarr
import psutil
from preference_inconsistency import PreferenceAnalyzer  # Import from our other module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Optimized configuration for streaming pipeline."""
    buffer_size: int = 10000
    batch_size: int = 32
    compression: str = 'lz4'
    storage_backend: str = 'hdf5'
    max_memory_gb: float = 4.0
    num_workers: int = 4
    enable_preference_analysis: bool = True
    retention_hours: int = 24

@dataclass
class ActivationBatch:
    """Optimized batch container with preference analysis support."""
    session_id: str
    layer_name: str
    timestamps: List[float]
    tensors: List[Union[torch.Tensor, np.ndarray]]
    metadata: List[Dict]
    batch_id: str = None
    preference_scores: Optional[Dict] = None
    
    def __post_init__(self):
        self.batch_id = self.batch_id or str(uuid.uuid4())

class CompressionEngine:
    """Optimized compression handler with LZ4 as default."""
    
    @staticmethod
    def compress(data: bytes, method: str = 'lz4') -> bytes:
        if method == 'lz4':
            return lz4.frame.compress(data)
        raise ValueError(f"Unsupported compression: {method}")
    
    @staticmethod
    def decompress(data: bytes, method: str = 'lz4') -> bytes:
        if method == 'lz4':
            return lz4.frame.decompress(data)
        raise ValueError(f"Unsupported compression: {method}")

class MemoryManager:
    """Optimized memory management with proactive cleaning."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_bytes = max_memory_gb * 1024**3
        self.cleanup_handlers = []
        
    def check_memory(self) -> bool:
        return psutil.Process().memory_info().rss < self.max_bytes
    
    def register_cleanup(self, handler: Callable):
        self.cleanup_handlers.append(handler)
    
    def perform_cleanup(self):
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

class StreamingBuffer:
    """High-performance thread-safe buffer."""
    
    def __init__(self, maxsize: int = 10000):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.stats = defaultdict(int)
    
    def put(self, item) -> bool:
        with self.lock:
            if len(self.buffer) >= self.buffer.maxlen:
                self.stats['overflows'] += 1
                return False
            self.buffer.append(item)
            self.stats['added'] += 1
            self.not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[object]:
        with self.not_empty:
            if not self.buffer:
                if not self.not_empty.wait(timeout or None):
                    return None
            return self.buffer.popleft() if self.buffer else None

class StorageBackend:
    """Unified storage interface with preference support."""
    
    def store_batch(self, batch: ActivationBatch) -> str:
        raise NotImplementedError
    
    def load_batch(self, location: str) -> ActivationBatch:
        raise NotImplementedError

class HDF5Backend(StorageBackend):
    """Optimized HDF5 storage with metadata compression."""
    
    def __init__(self, base_path: str = "./activations"):
        self.base_path = Path(base_path).mkdir(exist_ok=True)
    
    def store_batch(self, batch: ActivationBatch) -> str:
        filepath = self.base_path / f"{batch.batch_id}.h5"
        with h5py.File(filepath, 'w') as f:
            # Store core data
            f.create_dataset('timestamps', data=np.array(batch.timestamps))
            tensors = f.create_group('tensors')
            for i, t in enumerate(batch.tensors):
                tensors.create_dataset(f't_{i}', data=np.array(t), compression='lz4')
            
            # Store metadata and preferences
            f.attrs.update({
                'session_id': batch.session_id,
                'layer_name': batch.layer_name,
                'metadata': json.dumps(batch.metadata),
                'preferences': json.dumps(batch.preference_scores or {})
            })
        return str(filepath)
    
    def load_batch(self, location: str) -> ActivationBatch:
        with h5py.File(location, 'r') as f:
            return ActivationBatch(
                session_id=f.attrs['session_id'],
                layer_name=f.attrs['layer_name'],
                timestamps=f['timestamps'][:].tolist(),
                tensors=[t[:] for t in f['tensors'].values()],
                metadata=json.loads(f.attrs['metadata']),
                preference_scores=json.loads(f.attrs.get('preferences', '{}')),
                batch_id=Path(location).stem
            )

class ActivationAnalyzer:
    """Unified analyzer for both safety and preference inconsistencies."""
    
    def __init__(self):
        self.safety_analyzer = SafetyAnalyzer()
        self.preference_analyzer = PreferenceAnalyzer()
    
    def analyze(self, batch: ActivationBatch) -> Dict:
        results = {
            'safety': self.safety_analyzer.analyze(batch),
            'preferences': self.preference_analyzer.detect_inconsistencies(batch.metadata)
        }
        batch.preference_scores = results['preferences']
        return results

class StreamProcessor:
    """Optimized processor with preference analysis integration."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.buffer = StreamingBuffer(config.buffer_size)
        self.memory = MemoryManager(config.max_memory_gb)
        self.storage = HDF5Backend()
        self.analyzer = ActivationAnalyzer()
        self.executor = ThreadPoolExecutor(config.num_workers)
        self.running = False
        
        # Setup memory management
        self.memory.register_cleanup(self._cleanup)
    
    def start(self):
        self.running = True
        for _ in range(self.config.num_workers):
            self.executor.submit(self._worker)
    
    def stop(self):
        self.running = False
        self.executor.shutdown()
    
    def submit(self, session_id: str, layer_name: str, 
               tensor: Union[torch.Tensor, np.ndarray], 
               metadata: Dict = None) -> bool:
        return self.buffer.put({
            'session_id': session_id,
            'layer_name': layer_name,
            'tensor': tensor,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
    
    def _worker(self):
        batch = defaultdict(list)
        while self.running:
            if not self.memory.check_memory():
                self.memory.perform_cleanup()
                time.sleep(0.1)
                continue
            
            data = self.buffer.get(timeout=1.0)
            if not data:
                continue
            
            key = (data['session_id'], data['layer_name'])
            batch[key].append(data)
            
            if len(batch[key]) >= self.config.batch_size:
                self._process_batch(batch[key])
                batch[key].clear()
    
    def _process_batch(self, data: List[Dict]):
        batch = ActivationBatch(
            session_id=data[0]['session_id'],
            layer_name=data[0]['layer_name'],
            timestamps=[d['timestamp'] for d in data],
            tensors=[d['tensor'] for d in data],
            metadata=[d['metadata'] for d in data]
        )
        
        try:
            # Perform unified analysis
            analysis = self.analyzer.analyze(batch)
            
            # Store with analysis results
            self.storage.store_batch(batch)
            
            logger.info(f"Processed batch {batch.batch_id} with {len(data)} items")
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

    def _cleanup(self):
        """Cleanup old data when memory is low."""
        cutoff = time.time() - (self.config.retention_hours * 3600)
        for f in Path("./activations").glob("*.h5"):
            if f.stat().st_mtime < cutoff:
                f.unlink()

async def demo_integrated():
    """Demo showing integration with preference system."""
    config = StreamingConfig(
        buffer_size=1000,
        batch_size=16,
        enable_preference_analysis=True
    )
    
    processor = StreamProcessor(config)
    processor.start()
    
    try:
        # Simulate data with preference metadata
        for i in range(100):
            tensor = torch.randn(32, 768)
            metadata = {
                'preferences': {
                    'choice': 'option_a' if i % 2 else 'option_b',
                    'confidence': np.random.uniform(0.7, 0.9),
                    'context': 'individual' if i % 3 else 'group'
                }
            }
            
            processor.submit(
                session_id="demo_session",
                layer_name=f"layer_{i%5}",
                tensor=tensor,
                metadata=metadata
            )
            await asyncio.sleep(0.01)
        
        await asyncio.sleep(2)  # Allow processing to complete
    finally:
        processor.stop()

if __name__ == "__main__":
    asyncio.run(demo_integrated())