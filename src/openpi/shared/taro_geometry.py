"""Taro SE(3) transforms; row rot6d internally, column rot6d on the wire.

Conventions follow Isaac-GR00T gr00t/data/state_action/rot6d.py and pose.py
(NVIDIA, Apache-2.0). No GR00T runtime dependency is required.
"""

import numpy as np


def rotation_matrix(value: np.ndarray, *, columns: bool = False) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6 or not np.isfinite(value).all():
        raise ValueError("Expected finite rot6d")
    first, second = value[..., :3], value[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("Degenerate first rotation axis")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("Degenerate second rotation axis")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-1 if columns else -2)


def rot6d(matrix: np.ndarray, *, columns: bool = False) -> np.ndarray:
    matrix = np.asarray(matrix)
    return (np.swapaxes(matrix, -1, -2) if columns else matrix)[..., :2, :].reshape(*matrix.shape[:-2], 6)


def change_convention(value: np.ndarray, *, to_columns: bool) -> np.ndarray:
    return rot6d(rotation_matrix(value, columns=not to_columns), columns=to_columns).astype(np.float32)


def relative_actions(state: np.ndarray, actions: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode absolute row-rot6d targets relative to current right EEF."""
    state, actions, mask = np.asarray(state), np.asarray(actions), np.asarray(mask, dtype=bool).copy()
    if state.shape[-1] != 38 or actions.shape[-1] != 29 or actions.shape != mask.shape:
        raise ValueError("Taro requires state 38, actions/mask (..., H, 29)")
    if not np.isfinite(state).all() or not np.isfinite(actions[mask]).all():
        raise ValueError("Nonfinite supervised Taro coordinate")
    mask[..., :9] = mask[..., :9].all(axis=-1, keepdims=True)
    safe = np.where(mask, actions, 0).astype(np.float64)
    safe[..., 3:9] = np.where(mask[..., 3:9], safe[..., 3:9], [1, 0, 0, 0, 1, 0])
    anchor = rotation_matrix(state[..., 12:18])
    inverse = np.swapaxes(anchor, -1, -2)
    safe[..., :3] = np.einsum("...ij,...hj->...hi", inverse, safe[..., :3] - state[..., None, 9:12])
    safe[..., 3:9] = rot6d(inverse[..., None, :, :] @ rotation_matrix(safe[..., 3:9]))
    return np.where(mask, safe, 0).astype(np.float32), mask


def absolute_actions(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Decode relative actions against the exact observation used for inference."""
    state, result = np.asarray(state), np.asarray(actions, dtype=np.float64).copy()
    if state.shape != (38,) or result.ndim != 2 or result.shape[-1] != 29:
        raise ValueError("Taro decode requires state (38,) and actions (H,29)")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predicted action")
    anchor = rotation_matrix(state[12:18])
    result[:, :3] = result[:, :3] @ anchor.T + state[9:12]
    result[:, 3:9] = rot6d(anchor @ rotation_matrix(result[:, 3:9]))
    return result.astype(np.float32)
