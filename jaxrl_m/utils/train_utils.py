from collections.abc import Mapping

import imageio
import numpy as np
import tensorflow as tf
import wandb
import jax.numpy as jnp


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        if isinstance(batches[0][key], Mapping):
            # to concatenate batch["observations"]["image"], etc.
            concatenated[key] = concatenate_batches([batch[key] for batch in batches])
        else:
            first_value = batches[0][key]
            assert all(batch[key].dtype == first_value.dtype for batch in batches)

            concatenated[key] = np.concatenate(
                [batch[key] for batch in batches], axis=0
            ).astype(first_value.dtype)
    return concatenated


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        if isinstance(batch[key], Mapping):
            # to index into batch["observations"]["image"], etc.
            indexed[key] = index_batch(batch[key], indices)
        else:
            indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    # indices = np.random.randint(batch["rewards"].shape[0], size=size)
    indices = np.random.choice(batch["rewards"].shape[0], size=size, replace=False)
    return index_batch(batch, indices)


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def preprocess_action(actions, action_dim=7):
    """Safe: works even if 'actions' is shared inside a PyTree."""
    if actions.ndim == 3:
        # Slice last dimension
        truncated = actions[..., :action_dim]
        B, H, D = truncated.shape
        flat = truncated.reshape(B, H * D)
        return flat
    return actions  # unchanged


def repack_action(actions, action_dim=7, pad=False, pad_dim=32):
    """Safe repack: no aliasing, no mutation on shared arrays."""
    out = actions

    if out.ndim == 2:
        B, H_D = out.shape
        D = action_dim
        H = H_D // D
        out = out.reshape(B, H, D)

    if pad and out.shape[-1] != pad_dim:
        out = pad_to_dim(out, pad_dim, axis=-1)

    return out


def pad_to_dim(x, target_dim: int, axis: int = -1, value: float = 0.0):
    """Pad an array to the target dimension along a given axis."""
    current_dim = x.shape[axis]
    if current_dim >= target_dim:
        return x

    pad_amount = target_dim - current_dim
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_amount)

    # Backend-agnostic padding
    if isinstance(x, jnp.ndarray):
        return jnp.pad(x, pad_width, mode='constant', constant_values=value)
    else:
        return np.pad(x, pad_width, mode='constant', constant_values=value)

