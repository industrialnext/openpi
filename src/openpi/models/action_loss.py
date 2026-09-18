"""Flow loss reduction with per-sample physical supervision counts."""

import jax.numpy as jnp


def masked_flow_loss(prediction, target, mask=None):
    squared = jnp.square(prediction - target)
    if mask is None:
        return jnp.mean(squared, axis=-1)
    count = jnp.sum(mask, axis=(-2, -1))
    # The trainer averages over samples and horizon, so multiply by horizon here.
    return prediction.shape[-2] * jnp.sum(jnp.where(mask, squared, 0), axis=-1) / jnp.maximum(count[..., None], 1)
