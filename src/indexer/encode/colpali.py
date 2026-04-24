"""ColPali encoder wrapper for the indexing pipeline."""

from __future__ import annotations

from typing import Any

from PIL import Image

from indexer.shared.errors import DependencyUnavailableError, IndexingRuntimeError

try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch.
    torch = None


def _build_model_load_kwargs(device: str, torch_dtype: Any) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device != "cpu":
        load_kwargs["device_map"] = device
    return load_kwargs


def _resolve_torch_dtype(device: str) -> Any:
    if torch is None:
        raise DependencyUnavailableError(
            "torch is required for ColPali encoding. Install 'torch'."
        )
    if device == "cuda" and hasattr(torch, "bfloat16"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _infer_embedding_dimension(model: Any) -> int:
    dim = getattr(model, "dim", None)
    if isinstance(dim, int):
        return dim

    custom_text_proj = getattr(model, "custom_text_proj", None)
    out_features = getattr(custom_text_proj, "out_features", None)
    if isinstance(out_features, int):
        return out_features

    raise IndexingRuntimeError("Unable to determine embedding dimension for the loaded model.")


def _load_model_classes(model_name: str) -> tuple[Any, Any]:
    try:
        from colpali_engine.models import (
            ColPali,
            ColPaliProcessor,
            ColQwen2,
            ColQwen2_5,
            ColQwen2_5_Processor,
            ColQwen2Processor,
        )
    except ImportError as error:  # pragma: no cover - external dependency.
        raise DependencyUnavailableError(
            "colpali-engine is required for ColPali encoding."
        ) from error

    normalized_model_name = model_name.lower()
    if "colqwen2.5" in normalized_model_name or "colqwen2_5" in normalized_model_name:
        return ColQwen2_5, ColQwen2_5_Processor
    if "colqwen2" in normalized_model_name:
        return ColQwen2, ColQwen2Processor
    return ColPali, ColPaliProcessor


def _prepare_inputs(processor: Any, image: Image.Image) -> Any:
    if hasattr(processor, "process_images"):
        return processor.process_images([image])
    return processor(images=image, return_tensors="pt")


def _prepare_query_inputs(processor: Any, query_text: str) -> Any:
    if hasattr(processor, "process_queries"):
        return processor.process_queries([query_text])
    return processor(text=query_text, return_tensors="pt")


class ColPaliPageEncoder:
    """Encodes rendered pages into ColPali patch embeddings."""

    def __init__(self, model_name: str, device: str) -> None:
        if torch is None:
            raise DependencyUnavailableError(
                "torch is required for ColPali encoding. Install 'torch'."
            )

        self._device = device
        model_class, processor_class = _load_model_classes(model_name)
        self._processor = processor_class.from_pretrained(model_name)

        torch_dtype = _resolve_torch_dtype(device)
        load_kwargs = _build_model_load_kwargs(device=device, torch_dtype=torch_dtype)
        self._model = model_class.from_pretrained(model_name, **load_kwargs)
        if "device_map" not in load_kwargs:
            self._model = self._model.to(device)
        self._model = self._model.eval()
        self.embedding_dimension: int = _infer_embedding_dimension(self._model)

    def encode_page(self, image: Image.Image) -> list[list[float]]:
        """Encode one rendered page into patch embeddings."""

        try:
            inputs = _prepare_inputs(self._processor, image)
            inputs = _move_inputs_to_device(inputs, self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            page_tensor = self._extract_page_tensor(outputs)
            return page_tensor.detach().cpu().float().tolist()
        except Exception as error:  # pragma: no cover - depends on model/runtime.
            raise IndexingRuntimeError(f"Failed to encode page with ColPali: {error}") from error

    def encode_query(self, query_text: str) -> list[list[float]]:
        """Encode one text query into late-interaction embeddings."""

        if not query_text.strip():
            raise IndexingRuntimeError("Query text must not be blank.")

        try:
            inputs = _prepare_query_inputs(self._processor, query_text)
            inputs = _move_inputs_to_device(inputs, self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            query_tensor = self._extract_page_tensor(outputs)
            return query_tensor.detach().cpu().float().tolist()
        except Exception as error:  # pragma: no cover - depends on model/runtime.
            raise IndexingRuntimeError(f"Failed to encode query with ColPali: {error}") from error

    def score_query_to_pages(
        self,
        query_embeddings: list[list[float]],
        page_embeddings: list[list[list[float]]],
    ) -> list[float]:
        """Score one query embedding set against page embedding candidates."""

        if not query_embeddings:
            raise IndexingRuntimeError("Query embeddings must not be empty.")
        if not page_embeddings:
            raise IndexingRuntimeError("Page embeddings must not be empty.")

        try:
            query_tensors = [torch.tensor(query_embeddings, dtype=torch.float32)]
            page_tensors = [
                torch.tensor(candidate_embeddings, dtype=torch.float32)
                for candidate_embeddings in page_embeddings
            ]
            score_function = getattr(self._processor, "score_multi_vector", None)
            if score_function is None:
                raise IndexingRuntimeError("Processor does not expose score_multi_vector.")
            scores = score_function(query_tensors, page_tensors, device=self._device)
            return scores[0].detach().cpu().float().tolist()
        except Exception as error:  # pragma: no cover - depends on model/runtime.
            raise IndexingRuntimeError(f"Failed to score query against pages: {error}") from error

    @staticmethod
    def _extract_page_tensor(outputs: Any) -> Any:
        if hasattr(outputs, "embeddings"):
            tensor = outputs.embeddings
        elif hasattr(outputs, "last_hidden_state"):
            tensor = outputs.last_hidden_state
        else:
            tensor = outputs

        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if getattr(tensor, "ndim", None) == 3:
            return tensor[0]
        if getattr(tensor, "ndim", None) == 2:
            return tensor
        raise IndexingRuntimeError("Unexpected ColPali output shape for page embeddings.")
