from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import os
import numpy as np
import tensorflow as tf
from absl import logging

import torch
import clip

from jaxrl_m.data.tf_augmentations import augment as augment_fn
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS
import json
from openpi.training.config import get_config
import openpi.transforms as _transforms
import openpi.models.model as _model
import jax


def load_language_embeddings_str(path: str) -> dict[int, np.ndarray]:
    """
    Loads language embeddings from a JSON file and converts them to float32 numpy arrays.

    Returns:
        Dict[int, np.ndarray] mapping task_id → embedding (shape [512], dtype float32)
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Convert keys to int and lists to np.float32 arrays
    embeddings = {
        str(k): np.array(v, dtype=np.float32)
        for k, v in data.items()
    }

    # Optionally verify dimensions
    for k, v in embeddings.items():
        assert v.shape == (512,), f"Task {k} has wrong shape {v.shape}"

    return embeddings

class ImageReplayBufferPi:
    def __init__(
        self,
        data_paths: List[str],
        seed: int,
        env_name: str, 
        goal_relabeling_strategy: Optional[str] = None,
        goal_relabeling_kwargs: dict = {},
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        train: bool = True,
        tfrecords_include_next_observations: bool = True,
        states_only: bool = False,
        augment: bool = False,
        augment_kwargs: dict = {},
        include_next_actions: bool = False,
        use_language: bool = True,
        use_wrist_view: bool = True,
        config=None,
    ):
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.is_train = train
        self.cache = cache
        self.tfrecords_include_next_observations = tfrecords_include_next_observations
        self.states_only = states_only
        self.augment = augment
        self.augment_kwargs = augment_kwargs
        self.include_next_actions = include_next_actions
        self.use_language = use_language
        self.use_wrist_view = use_wrist_view
        self.env_name = env_name
        if self.env_name == "calvin":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=device)
            self._clip_model.eval()
            self._clip_device = device
        elif self.env_name == "libero":
            self.lang2embedding = load_language_embeddings_str("data_info/libero_language2embeddings_normalised.json")
        else:
            raise NotImplementedError

        self.config = config
        # Data configs, setup and utils
        self.data_config = config.data.create(config.assets_dirs, config.model)
        self.data_norm_stats = self.data_config.norm_stats
        self.data_transforms = [*self.data_config.repack_transforms.inputs,
            *self.data_config.data_transforms.inputs,
            _transforms.Normalize(self.data_norm_stats, use_quantiles=self.data_config.use_quantile_norm),
            *self.data_config.model_transforms.inputs,]
        self.data_transforms = _transforms.compose(self.data_transforms)

        dataset = self._construct_tf_dataset(data_paths, seed)

        if train:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=seed).repeat()

            # if augment:
            #     dataset = dataset.enumerate(start=seed)
            #     dataset = dataset.map(
            #         self._augment, num_parallel_calls=tf.data.AUTOTUNE
            #     )

        # dataset = dataset.batch(
        #     batch_size,
        #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
        #     drop_remainder=True,
        #     deterministic=not train,
        # )

        self.tf_dataset = dataset



    def _construct_tf_dataset(
        self, data_paths: List[str], seed: int
    ) -> tf.data.Dataset:
        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(data_paths).shuffle(
            len(data_paths), seed
        )

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # cache before add_goals because add_goals introduces randomness
        if self.cache:
            dataset = dataset.cache()

        # dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        if self.states_only:
            dataset = dataset.map(
                lambda x: {
                    k: (
                        x[k]
                        if k not in ["observations", "next_observations", "goals"]
                        else x[k]["proprio"]
                    )
                    for k in x
                },
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        # "next_observations/images0": tf.uint8,
        # "next_observations/state": tf.float32,
        "actions": tf.float32,
        # "terminals": tf.bool,
        # "truncates": tf.bool,
    }

    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        if self.goal_relabeling_strategy is None:
            self.PROTO_TYPE_SPEC["rewards"] = tf.float32
            self.PROTO_TYPE_SPEC["masks"] = tf.float32
            self.PROTO_TYPE_SPEC["mc_returns"] = tf.float32
        if self.tfrecords_include_next_observations:
            self.PROTO_TYPE_SPEC["next_observations/images0"] = tf.uint8
            self.PROTO_TYPE_SPEC["next_observations/state"] = tf.float32
        if self.states_only:
            self.PROTO_TYPE_SPEC.pop("observations/images0")
            if self.tfrecords_include_next_observations:
                self.PROTO_TYPE_SPEC.pop("next_observations/images0")
        if self.use_language:
            self.PROTO_TYPE_SPEC["language"] = tf.string
        if self.use_wrist_view:
            self.PROTO_TYPE_SPEC["observations/images1"] = tf.uint8
            if self.tfrecords_include_next_observations:
                self.PROTO_TYPE_SPEC["next_observations/images1"] = tf.uint8
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        # parsed_tensors = {
        #     key: tf.io.parse_tensor(parsed_features[key], dtype)
        #     for key, dtype in self.PROTO_TYPE_SPEC.items()
        # }
        parsed_tensors = {}
        for key, dtype in self.PROTO_TYPE_SPEC.items():
            # If this feature is a plain string (like language text), don't parse it as a tensor
            if dtype == tf.string:
                parsed_tensors[key] = parsed_features[key]  # raw string bytes
            else:
                parsed_tensors[key] = tf.io.parse_tensor(parsed_features[key], dtype)

        if not self.tfrecords_include_next_observations:
            states = parsed_tensors["observations/state"]
            parsed_tensors["observations/state"] = states[:-1]
            parsed_tensors["next_observations/state"] = states[1:]
            # if self.states_only:
            #     parsed_tensors["observations/images0"] = None
            #     parsed_tensors["next_observations/images0"] = None
            # else:
            if not self.states_only:
                images = parsed_tensors["observations/images0"]
                parsed_tensors["observations/images0"] = images[:-1]
                parsed_tensors["next_observations/images0"] = images[1:]
                if self.use_wrist_view:
                    wrist_images = parsed_tensors["observations/images1"]
                    parsed_tensors["observations/images1"] = wrist_images[:-1]
                    parsed_tensors["next_observations/images1"] = wrist_images[1:]
        if self.use_language:

            # Repeat prompt for each timestep (tokenizer doesn't handle batching)
            length = tf.shape(parsed_tensors["observations/state"])[0]
            parsed_tensors["prompt"] = tf.repeat(parsed_tensors["language"][None], length, axis=0)
           
            parsed_tensors["language"] = tf.repeat(
                parsed_tensors["language"][None], length, axis=0
            )

            def _encode_clip(text_tensor):
                """Run CLIP text encoder and return a 512-D embedding (or) use a cached embedding dict."""

                # Convert TF string to Python str
                text_str = text_tensor.numpy().decode("utf-8")
                if self.env_name == "libero":
                    text_features = self.lang2embedding[text_str]
                elif self.env_name == "calvin":
                    tokens = clip.tokenize([text_str]).to(self._clip_device)
                    with torch.no_grad():
                        emb = self._clip_model.encode_text(tokens)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    text_features = emb.cpu().numpy().squeeze().astype("float32")
                else:
                    raise NotImplementedError
                return text_features

            clip_emb = tf.py_function(
                func=_encode_clip,
                inp=[parsed_tensors["language"][0]],
                Tout=tf.float32,
            )
            clip_emb.set_shape([512])  # ViT-B/32 output dim
            clip_emb = tf.repeat(clip_emb[None, :], length, axis=0)
            parsed_tensors["language_embedding"] = clip_emb
        if self.include_next_actions:
            # add the next action as part of the observation
           raise NotImplementedError
        # restructure the dictionary into the downstream format
        ah = self.config.model.action_horizon

        # core arrays (obs/state and next_obs/state already aligned)
        obs_state = parsed_tensors["observations/state"]              # [T]
        next_state = parsed_tensors["next_observations/state"]        # [T]
        actions = parsed_tensors["actions"]                           # [T, ad]

        T = tf.shape(obs_state)[0]

        # number of valid windows = T - (ah - 1)
        W = T - ah + 1
        start_idx = tf.range(W)   # [0,1,2,...,W-1]

        # ----- window states -----
        obs_state = tf.gather(obs_state, start_idx)
        next_state = tf.gather(next_state, start_idx)

        # ----- window images -----
        obs_images = {}
        next_images = {}
        if not self.states_only:
            imgs0 = parsed_tensors["observations/images0"]
            nimgs0 = parsed_tensors["next_observations/images0"]

            obs_images["image"] = tf.gather(imgs0, start_idx)
            next_images["image"] = tf.gather(nimgs0, start_idx)

            if self.use_wrist_view:
                imgs1 = parsed_tensors["observations/images1"]
                nimgs1 = parsed_tensors["next_observations/images1"]
                obs_images["wrist_image"] = tf.gather(imgs1, start_idx)
                next_images["wrist_image"] = tf.gather(nimgs1, start_idx)

        # ----- window actions: [W, ah, action_dim] -----
        actions_window = tf.map_fn(
            lambda t: actions[t : t + ah],
            start_idx,
            fn_output_signature=tf.float32,
        )
        return {
        "observations": {
            "proprio": obs_state,
            **obs_images,
            **({"language": tf.gather(parsed_tensors["language_embedding"], start_idx)}
               if "language_embedding" in parsed_tensors else {})
        },
        "next_observations": {
            "proprio": next_state,
            **next_images,
        },
        "actions": actions_window,
        **({"next_actions": tf.gather(parsed_tensors["next_actions"], start_idx)}
           if self.include_next_actions else {}),
        "terminals": tf.zeros([W], dtype=tf.bool),
        "truncates": tf.zeros([W], dtype=tf.bool),
        **({
            "rewards": tf.gather(parsed_tensors["rewards"], start_idx),
            "masks": tf.gather(parsed_tensors["masks"], start_idx),
            "mc_returns": tf.gather(parsed_tensors["mc_returns"], start_idx),
        } if "rewards" in parsed_tensors else {}),
        **({"prompt": tf.gather(parsed_tensors["prompt"], start_idx)}
           if "prompt" in parsed_tensors else {}),
    }
 
    def iterator(self, batch_size, training=True):
        tf_iter = (
            self.tf_dataset.batch(
                batch_size,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                drop_remainder=True,
                deterministic=not self.is_train,
            )
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

        for batch in tf_iter:
            flat = {}

            # observations
            for k, v in batch["observations"].items():
                if k != "language":
                    flat["observation/" + k] = v

            # # # next_observations
            # for k, v in batch["next_observations"].items():
            #     flat["next_observation/" + k] = v

            # actions
            flat["actions"] = batch["actions"]

            # prompt (language)
            if "prompt" in batch:
                flat["prompt"] = batch["prompt"]
            
            # print("flat keys: ", flat.keys())

            output = self._apply_data_transforms(flat, training=training)

            yield output


    def _apply_data_transforms(self, batch, training=False):
        """
        Apply self.data_transforms (which expect single-sample dicts)
        across a batched dictionary.
        """

        B = next(iter(batch.values())).shape[0]

        # Split into per-sample dicts (zero copy views)
        samples = []
        for i in range(B):
            samples.append(
                jax.tree_util.tree_map(lambda x: x[i], batch)
            )

        for s in samples:

            # ---- Fix prompt decoding ----
            if "prompt" in s:
                p = s["prompt"]
                if isinstance(p, np.ndarray):
                    if p.dtype.type is np.bytes_ or (p.dtype == object and isinstance(p[0], bytes)):
                        s["prompt"] = np.array([x.decode("utf-8") for x in p], dtype=object)
                elif isinstance(p, bytes):
                    s["prompt"] = p.decode("utf-8")

            # ---- Fix proprio → state renaming ----
            if "observation/proprio" in s:
                s["observation/state"] = s.pop("observation/proprio")

            if "next_observation/proprio" in s:
                s["next_observation/state"] = s.pop("next_observation/proprio")
            if "actions" in s:
                s["actions"] = np.array(s["actions"], copy=True)
        # Apply your pre-defined CompositeTransform
        # print("s keys: ", samples[0].keys())
        transformed = [self.data_transforms(s) for s in samples]
        # print(transformed[0].keys())
        
        # Rebatch
        out = self.batch_stack(transformed)
        # print(out.keys())
        return out
        # if training:
        #     return _model.Observation.from_dict(out), out["actions"]
        # else:
        #     return _model.Observation.from_dict(out)
    
    def batch_stack(self, samples):
        return jax.tree_util.tree_map(
            lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
            *samples
        )





def save_trajectory_as_tfrecord(trajectory: Dict[str, np.ndarray], path: str):
    def tensor_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
        )

    assert path.endswith(".tfrecord")
    assert trajectory.keys() >= {
        "observations",
        "next_observations",
        "actions",
    }
    assert type(trajectory["observations"]) == list
    assert trajectory["observations"][0].keys() >= {"image", "proprio"}

    if tf.io.gfile.exists(path):
        print(f"Warning: Removing existing file at {path}")
        try:
            tf.io.gfile.remove(path)
        except Exception as e:
            breakpoint()

    tf.io.gfile.makedirs(os.path.dirname(path))

    with tf.io.TFRecordWriter(path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "observations/images0": tensor_feature(
                        np.array(
                            [o["image"] for o in trajectory["observations"]],
                            dtype=np.uint8,
                        )
                    ),
                    "observations/state": tensor_feature(
                        np.array(
                            [o["proprio"] for o in trajectory["observations"]],
                            dtype=np.float32,
                        )
                    ),
                    "next_observations/images0": tensor_feature(
                        np.array(
                            [o["image"] for o in trajectory["next_observations"]],
                            dtype=np.uint8,
                        )
                    ),
                    "next_observations/state": tensor_feature(
                        np.array(
                            [o["proprio"] for o in trajectory["next_observations"]],
                            dtype=np.float32,
                        )
                    ),
                    "actions": tensor_feature(
                        np.array(trajectory["actions"], dtype=np.float32)
                    ),
                    **(
                        {
                            "rewards": tensor_feature(
                                np.array(trajectory["rewards"], dtype=np.float32)
                            ),
                            "masks": tensor_feature(
                                np.array(trajectory["masks"], dtype=np.float32)
                            ),
                            "mc_returns": tensor_feature(
                                np.array(trajectory["mc_returns"], dtype=np.float32)
                            ),
                        }
                        if "rewards" in trajectory
                        else {}
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())

if __name__ == "__main__":
    import os
    import glob
    import numpy as np

    # -------------------------------
    # Load config
    # -------------------------------
    config = get_config("pi0_fast_libero_low_mem_finetune_custom")
    print("Assets dirs:", config.assets_dirs)
    print("Model config:", config.model)

    data_config = config.data.create(config.assets_dirs, config.model)
    print("data config:", data_config)

    # -------------------------------
    # Locate TFRecords
    # -------------------------------
    tfrecord_dir = "/data/hf_cache/datasets/LIBERO/libero_10_tf"
    data_paths = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")))

    if not data_paths:
        raise FileNotFoundError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"[INFO] Found {len(data_paths)} TFRecord files.")
    print("[INFO] Using first 5 files for integration test.\n")

    # -------------------------------
    # Construct Replay Buffer
    # -------------------------------
    buffer = ImageReplayBufferPi(
        data_paths=data_paths[:5],
        seed=42,
        use_language=True,
        cache=False,
        tfrecords_include_next_observations=False,
        env_name="libero",
        config=config,
    )

    iterator = buffer.iterator(batch_size=8)

    print("[INFO] Fetching one batch...\n")

    # -------------------------------
    # Your iterator returns:
    #   (Observation, actions)
    # -------------------------------
    obs_struct, actions = next(iterator)

    print("========== BATCH STRUCTURE ==========\n")

    # print("Observation object:", obs_struct)
    print("Actions shape:", actions.shape, "\n")

    # ------------------------------------------------------
    # Images
    # ------------------------------------------------------
    print("------- IMAGE KEYS -------")
    print("Image cameras:", list(obs_struct.images.keys()), "\n")

    for cam, img in obs_struct.images.items():
        print(f"[{cam}] image shape:", img.shape)
        print(f"First pixel: {img[0,0,0]}")
        break

    # ------------------------------------------------------
    # Image Masks
    # ------------------------------------------------------
    if obs_struct.image_masks:
        print("\n------- IMAGE MASKS -------")
        print("Mask cameras:", list(obs_struct.image_masks.keys()))
        for cam, mask in obs_struct.image_masks.items():
            print(f"[{cam}] mask shape:", mask.shape)
            break

    # ------------------------------------------------------
    # State
    # ------------------------------------------------------
    print("\n------- STATE -------")
    print("State shape:", obs_struct.state.shape)
    print("State[0]    :", obs_struct.state[0])

    # ------------------------------------------------------
    # Tokenized Prompt
    # ------------------------------------------------------
    print("\n------- TOKENIZED PROMPT -------")
    if obs_struct.tokenized_prompt is not None:
        print("Tokenized prompt shape:", obs_struct.tokenized_prompt.shape)
        print("Prompt tokens [0,:10]:", obs_struct.tokenized_prompt[0, :10])
    else:
        print("No tokenized prompt in batch.")

    # ------------------------------------------------------
    # Token Mask
    # ------------------------------------------------------
    if obs_struct.tokenized_prompt_mask is not None:
        print("\nPrompt mask shape:", obs_struct.tokenized_prompt_mask.shape)
        print("Mask[0,:10]:", obs_struct.tokenized_prompt_mask[0, :10])

    # ------------------------------------------------------
    # Actions
    # ------------------------------------------------------
    print("\n------- ACTIONS -------")
    print("Actions shape:", actions.shape)
    print("First action:", actions[0])

    print("\n==========================================")
    print("   ✅ PI-0.5 ImageReplayBuffer test PASSED")
    print("==========================================\n")

