"""ColPali encoder wrapper for the indexing pipeline."""

from __future__ import annotations

from typing import Any

from PIL import Image

from indexer.shared.errors import DependencyUnavailableError, IndexingRuntimeError

try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch.
    torch = None


class ColPaliPageEncoder:
    """Encodes rendered pages into ColPali patch embeddings."""

    def __init__(self, model_name: str, device: str) -> None:
        if torch is None:
            raise DependencyUnavailableError(
                "torch is required for ColPali encoding. Install 'torch'."
            )

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as error:  # pragma: no cover - external dependency.
            raise DependencyUnavailableError(
                "colpali-engine is required for ColPali encoding."
            ) from error

        self._device = device
        self._processor = ColPaliProcessor.from_pretrained(model_name)

        torch_dtype = torch.float32
        if device == "cuda" and hasattr(torch, "bfloat16"):
            torch_dtype = torch.bfloat16

        self._model = ColPali.from_pretrained(model_name, torch_dtype=torch_dtype)
        self._model = self._model.to(device).eval()

    def encode_page(self, image: Image.Image) -> list[list[float]]:
        """Encode one rendered page into patch embeddings."""

        try:
            inputs = self._processor(images=image, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._device)
            else:
                inputs = {
                    key: value.to(self._device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }

            with torch.no_grad():
                outputs = self._model(**inputs)

            page_tensor = self._extract_page_tensor(outputs)
            return page_tensor.detach().cpu().float().tolist()
        except Exception as error:  # pragma: no cover - depends on model/runtime.
            raise IndexingRuntimeError(f"Failed to encode page with ColPali: {error}") from error

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
