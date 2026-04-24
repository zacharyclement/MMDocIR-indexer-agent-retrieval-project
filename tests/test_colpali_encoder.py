"""Tests for the ColPali encoder integration helpers."""

from types import SimpleNamespace

import pytest

from indexer.encode.colpali import (
    ColPaliPageEncoder,
    _build_model_load_kwargs,
    _infer_embedding_dimension,
)
from indexer.shared.errors import IndexingRuntimeError


class _FakeTensor:
    def __init__(self, ndim: int, first_value: object | None = None) -> None:
        self.ndim = ndim
        self._first_value = first_value

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self._first_value


def test_extract_page_tensor_uses_embeddings_field() -> None:
    tensor = _FakeTensor(ndim=2)
    outputs = SimpleNamespace(embeddings=tensor)

    result = ColPaliPageEncoder._extract_page_tensor(outputs)

    assert result is tensor


def test_extract_page_tensor_uses_last_hidden_state_first_batch() -> None:
    page_tensor = _FakeTensor(ndim=2)
    outputs = SimpleNamespace(last_hidden_state=_FakeTensor(ndim=3, first_value=page_tensor))

    result = ColPaliPageEncoder._extract_page_tensor(outputs)

    assert result is page_tensor


def test_extract_page_tensor_rejects_unexpected_shape() -> None:
    with pytest.raises(IndexingRuntimeError, match="Unexpected ColPali output shape"):
        ColPaliPageEncoder._extract_page_tensor(_FakeTensor(ndim=1))


def test_build_model_load_kwargs_uses_device_map_for_non_cpu() -> None:
    assert _build_model_load_kwargs(device="mps", torch_dtype="float32") == {
        "torch_dtype": "float32",
        "device_map": "mps",
    }


def test_build_model_load_kwargs_skips_device_map_for_cpu() -> None:
    assert _build_model_load_kwargs(device="cpu", torch_dtype="float32") == {
        "torch_dtype": "float32"
    }


def test_infer_embedding_dimension_uses_model_dim() -> None:
    assert _infer_embedding_dimension(SimpleNamespace(dim=320)) == 320


def test_infer_embedding_dimension_uses_projection_out_features() -> None:
    model = SimpleNamespace(custom_text_proj=SimpleNamespace(out_features=128))

    assert _infer_embedding_dimension(model) == 128
