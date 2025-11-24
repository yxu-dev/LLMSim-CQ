import numpy as np
import torch
from types import SimpleNamespace

from lm_eval.models.huggingface import HFLM


class _DummyHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        return None


def _write_minimal_codebooks(tmp_path):
    k = np.zeros((1, 2, 4), dtype=np.float32)
    v = np.zeros((1, 2, 4), dtype=np.float32)
    np.save(tmp_path / "k_centroids_fisher_layer0.npy", k)
    np.save(tmp_path / "v_centroids_fisher_layer0.npy", v)


def test_maybe_enable_cq(tmp_path):
    _write_minimal_codebooks(tmp_path)
    dummy_owner = object.__new__(HFLM)
    dummy_owner._model = _DummyHFModel()
    dummy_owner._cq_disable_handle = None
    dummy_owner._config = dummy_owner._model.config

    HFLM._maybe_enable_cq(dummy_owner, str(tmp_path), "layer")

    assert getattr(dummy_owner.model, "_cq_enabled", False)
    assert callable(dummy_owner._cq_disable_handle)
