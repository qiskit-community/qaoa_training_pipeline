"""Model loading utilities for lightweight inference without Lightning dependencies.

This module provides functions to load PyTorch models directly from checkpoints
without requiring Lightning infrastructure, making deployment simpler and more
portable.
"""

from __future__ import annotations

import json
import pathlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from qaoa_training_pipeline.inference.torch_backend.inference_model import InferenceModel
from qaoa_training_pipeline.inference.torch_backend.models import builders

# Re-exported for backwards compatibility; the canonical (torch-free) definitions
# now live in config_io so the ONNX runtime can load configs without torch.
from qaoa_training_pipeline.inference.config_io import CONFIG_FILENAME, load_config


@contextmanager
def cross_os_path_compat() -> Iterator[None]:
    """Temporarily alias the foreign-OS ``Path`` class to the local one.

    Why: Lightning checkpoints pickle ``hyper_parameters`` verbatim, which often
    contain ``PosixPath``/``WindowsPath`` instances. Unpickling one on the other
    OS raises ``NotImplementedError: cannot instantiate 'PosixPath' on your
    system``. Aliasing is scoped to the ``with`` block so no global state leaks.
    """
    if sys.platform == "win32":
        saved = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        try:
            yield
        finally:
            pathlib.PosixPath = saved
    else:
        saved = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
        try:
            yield
        finally:
            pathlib.WindowsPath = saved


def load_checkpoint_file(checkpoint_path: Path, device: str) -> Any:
    """Deserialize a checkpoint file, tolerant of cross-OS ``Path`` pickles.

    Tries the safe ``weights_only=True`` fast path first (no pickle, no Path
    instantiation). Falls back to a full pickle load under a scoped path-class
    alias for legacy Lightning checkpoints that carry non-tensor metadata.
    """
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        pass  # Fall through to legacy path — Lightning checkpoints need pickle.

    try:
        with cross_os_path_compat():
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to deserialize checkpoint {checkpoint_path}: {e}. "
            f"This can happen when a checkpoint pickled on one OS is loaded "
            f"on another (PosixPath/WindowsPath incompatibility) or when the "
            f"checkpoint file is corrupt."
        ) from e


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return a clean ``state_dict``, stripping the Lightning ``core.`` prefix."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        raw = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint structure: expected dict, got {type(checkpoint).__name__}"
        )

    return {
        (key[len("core.") :] if key.startswith("core.") else key): value
        for key, value in raw.items()
    }


def load_model_from_config(
    config: dict[str, Any],
    config_path: Path,
    device: str = "cpu",
    strict: bool = True,
) -> InferenceModel:
    """Load model architecture and weights from ``config``, wrapped in ``InferenceModel``.

    This function:
      1. Extracts model initialization parameters from the config.
      2. Instantiates the model architecture via the ``builders`` registry.
      3. Loads checkpoint weights (Lightning-friendly).
      4. Wraps in ``InferenceModel`` (inference-only, no training/eval methods).
      5. Moves to device and sets to eval mode.

    Args:
        config: Parsed model_config.json dictionary.
        config_path: Path to the bundle directory or config file — used to
            resolve a relative ``checkpoint`` field.
        device: Device to load model on ("cpu", "cuda", ...).
        strict: Whether to strictly enforce state-dict loading.

    Raises:
        ValueError: If the config is missing required fields.
        FileNotFoundError: If the checkpoint file doesn't exist.
        RuntimeError: If checkpoint loading fails.
    """
    if "model_init" not in config:
        raise ValueError("Config missing 'model_init' field")

    model_init = config["model_init"]
    model_type = str(model_init.get("model_type", "")).lower()
    if not model_type:
        raise ValueError("Config model_init missing 'model_type'")
    if "input_dim" not in model_init:
        raise ValueError("Config model_init missing 'input_dim'")
    if "output_dim" not in model_init:
        raise ValueError("Config model_init missing 'output_dim'")

    # Instantiate model architecture via the registry — full model_init flows
    # through so architecture-specific fields (embed_dim, num_layers, edge_dim,
    # timesteps, ...) reach the model constructor.
    if model_type not in builders:
        raise KeyError(
            f"No builder registered for model type {model_type!r}. "
            f"Registered: {sorted(builders)}"
        )
    model = builders[model_type](model_init)

    if "checkpoint" not in config:
        raise ValueError("Config missing 'checkpoint' field")

    config_dir = config_path if config_path.is_dir() else config_path.parent
    checkpoint_path = config_dir / str(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            f"Resolved from config-dir {config_dir} + config['checkpoint']={config['checkpoint']!r}."
        )

    checkpoint = load_checkpoint_file(checkpoint_path, device)

    try:
        state_dict = extract_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to extract state_dict from {checkpoint_path}: {e}") from e

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        raise RuntimeError(f"Failed to apply state_dict to {model_type} model: {e}") from e

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"State dict loading failed for {model_type}. "
            f"Missing keys: {missing_keys}, "
            f"Unexpected keys: {unexpected_keys}"
        )

    return InferenceModel(
        core_model=model,
        model_type=str(model_type),
        device=device,
    )
