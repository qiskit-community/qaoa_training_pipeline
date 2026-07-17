"""Lightweight predictor for QAOA models without Lightning dependencies.

This module provides a minimal predictor implementation that loads models
directly from checkpoints without requiring Lightning infrastructure, making
it ideal for deployment scenarios where minimal dependencies are desired.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
    category=UserWarning,
)

from qaoa_training_pipeline.inference.datamodule_utils import undo_gamma_rescale
from qaoa_training_pipeline.inference.feature_extractor import AIFeatureExtractor
from qaoa_training_pipeline.inference.torch_backend.model_loader import (
    load_config,
    load_model_from_config,
)
from qaoa_training_pipeline.inference.torch_backend.models import predictors


def denormalize_qaoa_params(
    qaoa_params_norm: torch.Tensor,
    scale: float = torch.pi / 2,
) -> torch.Tensor:
    """Denormalize QAOA parameters from ``[0, 1]`` range."""
    return qaoa_params_norm * scale


class LightweightQAOAPredictor:
    """Lightweight predictor for QAOA models without Lightning dependencies.

    - Loads models directly from checkpoints (no Lightning wrapper).
    - Supports all model architectures via the ``builders`` / ``predictors``
      registries in ``models.py``.
    - Handles feature extraction and denormalization.

    Example:
        predictor = LightweightQAOAPredictor(
            config_path=Path("model_configs/graph_neural_network/model_config.json"),
            device="cpu",
        )
        params = predictor.predict(cost_op)
    """

    def __init__(
        self,
        config_path: Path,
        device: str = "cpu",
        strict: bool = True,
    ) -> None:
        """Initialize predictor from a model config.

        Args:
            config_path: Path to a config directory or model_config.json file.
            device: Device for inference ("cpu", "cuda", "mps", ...).
            strict: Whether to strictly enforce checkpoint loading.

        Raises:
            FileNotFoundError: If the config or checkpoint files don't exist.
            RuntimeError: If model loading fails.
        """
        self.config_path = Path(config_path)
        self.device = torch.device(device)
        self.strict = bool(strict)

        self.config = load_config(self.config_path)
        self.model = load_model_from_config(
            config=self.config,
            config_path=self.config_path,
            device=str(self.device),
            strict=self.strict,
        )

        self.model_init = self.config.get("model_init", {})
        self.model_type = str(self.model_init.get("model_type", "")).lower()
        self.in_features = list(self.model_init.get("in_features", []))
        self.output_dim = int(self.model_init.get("output_dim", 0))

        # Feature-normalization stats — LEGACY configs carried these under
        # model_init["norm_stats"]; current configs put them at the top level
        # as `feature_normalization`. Accept either.
        norm_stats = (
            self.config.get("feature_normalization") or self.model_init.get("norm_stats") or {}
        )
        self.feature_extractor = AIFeatureExtractor(
            in_features=self.in_features,
            norm_stats=norm_stats,
        )

    def metadata(self) -> dict[str, Any]:
        """Return predictor metadata from the config."""
        metadata = dict(self.config)
        if "output_dim" not in metadata and "model_init" in metadata:
            metadata["output_dim"] = metadata["model_init"].get("output_dim")
        return metadata

    def predict(
        self,
        cost_op: SparsePauliOp,
        mixer: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        denormalize: bool | None = None,
    ) -> list[float]:
        """Predict QAOA parameters from a cost operator.

        Args:
            cost_op: Cost operator (SparsePauliOp).
            mixer: Mixer circuit (optional, currently unused).
            ansatz_circuit: Ansatz circuit (optional, currently unused).
            initial_state: Initial state circuit (optional, currently unused).
            denormalize: Whether to apply the config's output rescaling.
                ``None`` (default) uses the config's ``denormalize_output``
                flag; pass ``False`` to receive the raw model output (e.g.
                when the caller supplies its own rescaling hook).

        Returns:
            Predicted QAOA parameters ``[beta_1, ..., beta_p, gamma_1, ..., gamma_p]``.
        """
        x_vec, features = self.feature_extractor.extract_and_pack(
            cost_op=cost_op,
            device=self.device,
        )
        features["x"] = x_vec

        # Route through the registry — predictors[type](core_model, features)
        # picks the right calling convention for the architecture.
        if self.model_type not in predictors:
            raise KeyError(
                f"No predictor registered for model type {self.model_type!r}. "
                f"Registered: {sorted(predictors)}"
            )
        with torch.no_grad():
            prediction = predictors[self.model_type](self.model.core, features)

        # Denormalize if needed — caller override wins over the config default.
        apply_denorm = (
            bool(self.config.get("denormalize_output", True))
            if denormalize is None
            else denormalize
        )
        if apply_denorm:
            scale = float(self.config.get("output_scale", torch.pi / 2))
            prediction = denormalize_qaoa_params(prediction, scale=scale)
            # undo the gamma rescaling applied to targets
            p = self.output_dim // 2
            if p > 0:
                rescale_a = float(features["rescale_a"])
                prediction = undo_gamma_rescale(prediction, p=p, rescale_a=rescale_a)

        output = prediction.detach().cpu().reshape(-1).tolist()

        if self.output_dim > 0 and len(output) != self.output_dim:
            raise ValueError(
                f"Predicted {len(output)} values, expected {self.output_dim}. "
                f"Model output shape: {prediction.shape}"
            )

        return [float(value) for value in output]
