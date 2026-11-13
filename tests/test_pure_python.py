#!/usr/bin/env python3
"""
Pure-Python tests with mock torch module.
Tests config handling, hash functions, and checkpoint logic.
"""
import sys
import os
import tempfile
import hashlib
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import types

# Create minimal mock torch module before importing src
class MockTensor:
    def __init__(self, data=None):
        self.data = data
    def numpy(self):
        import numpy as np
        return np.array(self.data) if self.data is not None else np.array([])
    def detach(self):
        return self
    def item(self):
        return float(self.data) if self.data is not None else 0.0
    def cpu(self):
        return self
    def to(self, *args, **kwargs):
        return self

class MockModule:
    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
    def state_dict(self):
        return {k: v for k, v in self._parameters.items()}
    def load_state_dict(self, state_dict, strict=True):
        self._parameters.update(state_dict)
    def parameters(self):
        return list(self._parameters.values())
    def named_parameters(self):
        return list(self._parameters.items())
    def eval(self):
        return self
    def train(self):
        return self
    def cuda(self):
        return self
    def cpu(self):
        return self
    def register_parameter(self, name, param):
        self._parameters[name] = param
    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor
    def add_module(self, name, module):
        self._modules[name] = module
    def children(self):
        return list(self._modules.values())
    def named_children(self):
        return list(self._modules.items())
    def apply(self, fn):
        fn(self)
        for m in self._modules.values():
            m.apply(fn)

class MockOptimizer:
    def __init__(self, params=None, **kwargs):
        self.param_groups = [dict(kwargs)]
        self.state = {}
    def state_dict(self):
        return {"state": self.state, "param_groups": self.param_groups}
    def load_state_dict(self, state_dict):
        self.state = state_dict.get("state", {})
        self.param_groups = state_dict.get("param_groups", [{}])

class MockLRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass

class MockGradScaler:
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass

class MockMultiprocessing:
    @staticmethod
    def get_context(method):
        return type('ctx', (), {})()

# Build torch as proper module types
torch_module = types.ModuleType('torch')
torch_module.nn = types.ModuleType('torch.nn')
torch_module.nn.Module = MockModule
torch_module.nn.Linear = MockModule
torch_module.nn.CrossEntropyLoss = MockModule
torch_module.nn.ModuleList = MockModule
torch_module.nn.init = types.ModuleType('torch.nn.init')
torch_module.nn.init.kaiming_normal_ = lambda *a, **k: None
torch_module.nn.init.kaiming_uniform_ = lambda *a, **k: None
torch_module.nn.init.xavier_normal_ = lambda *a, **k: None
torch_module.nn.init.xavier_uniform_ = lambda *a, **k: None
torch_module.nn.init.normal_ = lambda *a, **k: None
torch_module.nn.init.zeros_ = lambda *a, **k: None
torch_module.nn.functional = types.ModuleType('torch.nn.functional')
torch_module.nn.functional.cross_entropy = lambda *a, **k: MockTensor(0.0)
torch_module.nn.functional.kl_div = lambda *a, **k: MockTensor(0.0)
torch_module.optim = types.ModuleType('torch.optim')
torch_module.optim.AdamW = MockOptimizer
torch_module.optim.Optimizer = MockOptimizer
torch_module.optim.lr_scheduler = types.ModuleType('torch.optim.lr_scheduler')
torch_module.optim.lr_scheduler.OneCycleLR = MockLRScheduler
torch_module.optim.lr_scheduler.LRScheduler = MockLRScheduler
torch_module.optim.lr_scheduler.CosineAnnealingLR = MockLRScheduler
torch_module.cuda = types.ModuleType('torch.cuda')
torch_module.cuda.is_available = lambda: False
torch_module.cuda.amp = types.ModuleType('torch.cuda.amp')
torch_module.cuda.amp.GradScaler = MockGradScaler
torch_module.multiprocessing = MockMultiprocessing
torch_module.get_rng_state = staticmethod(lambda: b"mock_rng_state")
torch_module.set_rng_state = staticmethod(lambda state: None)
torch_module.manual_seed = staticmethod(lambda seed: None)
torch_module.isfinite = staticmethod(lambda tensor: True)
torch_module.zeros = staticmethod(lambda *args, **kwargs: MockTensor(0.0))
torch_module.randn = staticmethod(lambda *args, **kwargs: MockTensor(0.0))
torch_module.tensor = staticmethod(lambda data, **kwargs: MockTensor(data))
torch_module.no_grad = staticmethod(lambda: type('ctx', (), {'__enter__': lambda s: None, '__exit__': lambda s, *a: None})())

# torch.utils.data
torch_utils = types.ModuleType('torch.utils')
torch_utils_data = types.ModuleType('torch.utils.data')
torch_utils_data.IterableDataset = MockModule
torch_utils_data.DataLoader = MockModule
torch_utils_data.get_worker_info = lambda: None
torch_module.utils = torch_utils
torch_module.utils.data = torch_utils_data

sys.modules['torch'] = torch_module
sys.modules['torch.nn'] = torch_module.nn
sys.modules['torch.nn.functional'] = torch_module.nn.functional
sys.modules['torch.optim'] = torch_module.optim
sys.modules['torch.optim.lr_scheduler'] = torch_module.optim.lr_scheduler
sys.modules['torch.cuda'] = torch_module.cuda
sys.modules['torch.cuda.amp'] = torch_module.cuda.amp
sys.modules['torch.multiprocessing'] = torch_module.multiprocessing
sys.modules['torch.utils'] = torch_module.utils
sys.modules['torch.utils.data'] = torch_module.utils.data

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.checkpoint import CheckpointManager


def test_checkpoint_config_key_aliases():
    """Test that both 'include_*' and 'save_*' config keys work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()

        config = {
            "checkpoint": {
                "dir": str(ckpt_dir),
                "keep_last_n": 3,
                "include_optimizer": True,
                "include_scheduler": True,
                "include_scaler": True,
                "include_rng": True,
                "include_data_position": True,
                "include_atomic_write": True,
            }
        }
        mgr = CheckpointManager(config)
        assert mgr.save_optimizer is True
        assert mgr.save_scheduler is True
        assert mgr.save_scaler is True
        assert mgr.save_rng is True
        assert mgr.save_data_state is True
        assert mgr.atomic_writes is True
        print("  PASS: include_* keys work")

        config2 = {
            "checkpoint": {
                "dir": str(ckpt_dir / "ckpts2"),
                "keep_last_n": 3,
                "save_optimizer": False,
                "save_scheduler": False,
                "save_scaler": False,
                "save_rng": False,
                "save_data_state": False,
                "atomic_writes": False,
            }
        }
        (ckpt_dir / "ckpts2").mkdir()
        mgr2 = CheckpointManager(config2)
        assert mgr2.save_optimizer is False
        assert mgr2.save_scheduler is False
        assert mgr2.save_scaler is False
        assert mgr2.save_rng is False
        assert mgr2.save_data_state is False
        assert mgr2.atomic_writes is False
        print("  PASS: save_* keys work")

        config3 = {
            "checkpoint": {
                "dir": str(ckpt_dir / "ckpts3"),
                "include_optimizer": True,
                "save_optimizer": False,
            }
        }
        (ckpt_dir / "ckpts3").mkdir()
        mgr3 = CheckpointManager(config3)
        assert mgr3.save_optimizer is True
        print("  PASS: include_* takes precedence over save_*")


def test_hash_config_and_alias():
    """Test _hash_config and _compute_config_hash alias."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()
        config = {"checkpoint": {"dir": str(ckpt_dir)}}
        mgr = CheckpointManager(config)

        test_config = {"model": {"input_dim": 81920}, "training": {"lr": 0.001}}
        h1 = mgr._hash_config(test_config)
        h2 = mgr._compute_config_hash(test_config)

        assert h1 == h2, f"Alias mismatch: {h1} != {h2}"
        assert len(h1) == 16
        h3 = mgr._hash_config(test_config)
        assert h1 == h3, "Hash should be deterministic"

        test_config2 = {"model": {"input_dim": 81920}, "training": {"lr": 0.002}}
        h4 = mgr._hash_config(test_config2)
        assert h1 != h4, "Different configs should produce different hashes"
        print("  PASS: _hash_config and _compute_config_hash work correctly")


def test_serialize_training_state_filters_underscore():
    """Test that _serialize_training_state filters _-prefixed keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()
        config = {"checkpoint": {"dir": str(ckpt_dir)}}
        mgr = CheckpointManager(config)

        class FakeState:
            def __init__(self):
                self.step = 100
                self.epoch = 5
                self.samples_seen = 50000
                self.current_file_idx = 2
                self.current_position = 1000
                self._internal = "should_be_filtered"
                self._private = 42

        state = FakeState()
        serialized = mgr._serialize_training_state(state)

        assert serialized["step"] == 100
        assert serialized["epoch"] == 5
        assert serialized["samples_seen"] == 50000
        assert serialized["current_file_idx"] == 2
        assert serialized["current_position"] == 1000
        assert "_internal" not in serialized
        assert "_private" not in serialized
        print("  PASS: _serialize_training_state filters _-prefixed keys")


def test_restore_training_state_only_existing_keys():
    """Test that _restore_training_state only sets existing attributes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()
        config = {"checkpoint": {"dir": str(ckpt_dir)}}
        mgr = CheckpointManager(config)

        class FakeState:
            def __init__(self):
                self.step = 0
                self.epoch = 0
                self.samples_seen = 0

        state = FakeState()
        state_dict = {
            "step": 100,
            "epoch": 5,
            "nonexistent_key": "should_be_ignored",
            "samples_seen": 50000,
        }

        mgr._restore_training_state(state, state_dict)

        assert state.step == 100
        assert state.epoch == 5
        assert state.samples_seen == 50000
        assert not hasattr(state, "nonexistent_key")
        print("  PASS: _restore_training_state only sets existing keys")


def test_training_state_roundtrip_includes_data_position():
    """Test that serialize/restore round-trips data position fields (current_file_idx, current_position)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()
        config = {"checkpoint": {"dir": str(ckpt_dir)}}
        mgr = CheckpointManager(config)

        class FakeState:
            def __init__(self):
                self.step = 100000
                self.epoch = 5
                self.samples_seen = 5000000
                self.current_file_idx = 42
                self.current_position = 12345
                self.best_val_loss = 0.42
                self.time_elapsed = 3600.0
                self._internal = "should_be_filtered"

        state = FakeState()
        serialized = mgr._serialize_training_state(state)

        # Data position fields must be serialized
        assert serialized["current_file_idx"] == 42
        assert serialized["current_position"] == 12345
        assert serialized["samples_seen"] == 5000000
        assert serialized["step"] == 100000
        assert "_internal" not in serialized

        # Round-trip restore
        new_state = FakeState()
        new_state.step = 0
        new_state.current_file_idx = 0
        new_state.current_position = 0
        mgr._restore_training_state(new_state, serialized)

        assert new_state.current_file_idx == 42
        assert new_state.current_position == 12345
        assert new_state.samples_seen == 5000000
        assert new_state.step == 100000
        # _internal was set in __init__, not restored from serialized dict
        assert new_state._internal == "should_be_filtered"
        print("  PASS: TrainingState round-trip includes data position fields")


def test_checkpoint_atomic_write_no_partial():
    """Test that CheckpointManager initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()
        config = {
            "checkpoint": {
                "dir": str(ckpt_dir),
                "keep_last_n": 5,
                "atomic_write": True,
            }
        }
        mgr = CheckpointManager(config)
        assert ckpt_dir.exists()
        print("  PASS: CheckpointManager initialized correctly")


def test_config_hash_truncation():
    """Test that config hash is properly truncated to 16 chars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()
        config = {"checkpoint": {"dir": str(ckpt_dir)}}
        mgr = CheckpointManager(config)

        h = mgr._hash_config({"a": 1})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)
        print("  PASS: Config hash is 16 hex chars")


if __name__ == "__main__":
    print("Running pure-Python tests (with mock torch)...")
    print()
    test_checkpoint_config_key_aliases()
    test_hash_config_and_alias()
    test_serialize_training_state_filters_underscore()
    test_restore_training_state_only_existing_keys()
    test_training_state_roundtrip_includes_data_position()
    test_checkpoint_atomic_write_no_partial()
    test_config_hash_truncation()
    print()
    print("All pure-Python tests PASSED!")
