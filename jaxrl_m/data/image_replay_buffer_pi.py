from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import os
import numpy as np
import tensorflow as tf

from absl import logging

import torch
import clip

import json
from openpi.training.config import get_config
print("Imports 1")
import openpi.transforms as _transforms
import jax
from PIL import Image
print("Imports 2")


### Debugging setup ###
def inspect_tfrecords():
    # TFRECORD_PATTERN = "/data/hf_cache/datasets/LIBERO/libero_10_tf/*.tfrecord"
    TFRECORD_PATTERN = "/home/skowshik/vla/codebase/PolicyAgnosticRL/SET_1_libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/seed_0/image_replay_buffer/episode_1.tfrecord"

    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/images1": tf.uint8,  # Assuming wrist view exists
        "observations/state": tf.float32,
        "actions": tf.float32,
        "language": tf.string,
        "rewards": tf.float32,
        "masks": tf.float32,
        "mc_returns": tf.float32,
        "terminals": tf.float32,
        "truncates": tf.float32,
    }

    # 1. Get files
    files = glob.glob(TFRECORD_PATTERN)
    files = sorted(files)[0]
    if not files:
        print(f"❌ No files found matching: {TFRECORD_PATTERN}")
        return
    
    # print(f"✅ Found {len(files)} files. Inspecting the first one: {os.path.basename(files[0])}\n")

    # 2. Create a basic dataset
    dataset = tf.data.TFRecordDataset(files)

    # 3. Take one example and inspect it
    for raw_record in dataset.take(1):
        print("--- RAW FEATURE KEYS FOUND IN PROTO ---")
        # Parse the raw Example proto object to see exactly what keys exist on disk
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        found_keys = list(example.features.feature.keys())
        for k in sorted(found_keys):
            print(f" • {k}")
        print("-" * 40 + "\n")

        print("--- DECODING TENSORS ---")
        
        # 4. Parse the tensors
        # We assume every feature is a FixedLenFeature of type string (bytes), 
        # which is how `tf.io.serialize_tensor` saves them.
        features_dict = {
            k: tf.io.FixedLenFeature([], tf.string)
            for k in found_keys
        }
        
        parsed_features = tf.io.parse_single_example(raw_record, features_dict)

        out = {}
        
        for key in sorted(found_keys):
            # 1. Get the raw bytes
            raw_bytes = parsed_features[key]
            
            # 2. Check if we have a known dtype for this key in PROTO_TYPE_SPEC
            dtype = PROTO_TYPE_SPEC.get(key)
            
            try:
                if dtype is None:
                    # If we don't know the dtype, we can't parse the tensor, just print bytes info
                    print(f"❓ [{key}]: Unknown dtype in spec. Raw bytes len: {len(raw_bytes.numpy())}")
                elif dtype == tf.string:
                    # Strings are usually just stored directly
                    val = raw_bytes.numpy()
                    print(f"📝 [{key}] (String): {val}")
                else:
                    # It's a serialized tensor
                    tensor = tf.io.parse_tensor(raw_bytes, out_type=dtype)

                    out[key] = tensor
                    if tensor is None:
                        print(f"⚠️  [{key}]: Parse failed (returned None). Check dtype.")
                    else:
                        print(f"📦 [{key}]: Shape={tuple(tensor.shape)}, Dtype={tensor.dtype.name}")
                        
                        # logic to print stats if it helps
                        if key == "observations/state":
                            print(f"    -> First step values: {tensor[0].numpy()}")
                        elif "images" in key:
                            print(f"    -> Range: [{np.min(tensor)}, {np.max(tensor)}]")

            except Exception as e:
                print(f"❌ [{key}]: Error decoding - {e}")
            
            # breakpoint()
        
        breakpoint()

        # libero tfrecords spec
        # actions: (T, 7)
        # masks: (T,)
        # mc_returns: (T,)
        # observations/images0: (T + 1, 224, 224, 3)
        # observations/images1: (T + 1, 224, 224, 3)
        # observations/state: (T + 1, 15)
        # rewards: (T,): These are final timestep=1.0 reward and everything else decayed by 1.0 till start;
        #   reward for final timestep: gamma; Then it goes like gamma^2, gamma^3, ... gamma^(T-1) till start;
        # masks: always seem to be 1.0 till start;
        # mc_returns: discount_cumsum(rewards)
        

def save_image_tensor_as_png(tensor, path: str):
    """Saves a single image tensor as PNG."""
    # Convert TensorFlow tensor → NumPy
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()

    img = tensor

    # Handle batch dimension if present
    if img.ndim == 4:
        img = img[0]  # take first image in batch

    # Ensure float32 converted properly
    if img.dtype != np.uint8:
        # If float in [0,1], scale to [0,255]
        if img.max() <= 1.0:
            img = (img * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Remove any mask dims if accidentally passed
    if img.ndim != 3:
        raise ValueError(f"Image must be [H,W,3], got shape {img.shape}")

    # Convert and save
    Image.fromarray(img).save(path)
    print(f"Saved image to {path}")


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
        task_name: Optional[str] = None,
        use_8D=True,
        final_step_sparse_reward: bool = True, # Keeps only the last step of the trajectory for reward computation
        discount: float = 0.99,
        filter_successful_trajectories: bool = False,
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
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda:0" if torch.cuda.is_available() else "cpu" # For multi-gpu case
        self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=device)
        self._clip_model.eval()
        self._clip_device = device

        self.config = config
        # Data configs, setup and utils
        self.data_config = config.data.create(config.assets_dirs, config.model)
        self.data_norm_stats = self.data_config.norm_stats
        self.data_transforms = [*self.data_config.repack_transforms.inputs,
            *self.data_config.data_transforms.inputs,
            _transforms.Normalize(self.data_norm_stats, use_quantiles=self.data_config.use_quantile_norm),
            *self.data_config.model_transforms.inputs,]
        # breakpoint()
        self.data_transforms = _transforms.compose(self.data_transforms)
        self.task_name = task_name
        self.use_8D = use_8D
        self.final_step_sparse_reward = final_step_sparse_reward
        self.discount = discount
        self.filter_successful_trajectories = filter_successful_trajectories
        
        dataset = self._construct_tf_dataset(data_paths, seed)

        self.train = train

        # if train:
        #     dataset = dataset.repeat()
        #     if self.task_name is None:
        #         dataset = dataset.shuffle(512, reshuffle_each_iteration=True)


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
        # dataset = tf.data.Dataset.from_tensor_slices(data_paths).shuffle(
        #     len(data_paths), seed, reshuffle_each_iteration=True
        # )

        data_paths = sorted(data_paths)
        files = tf.data.Dataset.from_tensor_slices(data_paths)
        
        if self.is_train:
            files = files.shuffle(
                len(data_paths), seed=seed, reshuffle_each_iteration=True
            )

        # Interleave: each filename -> dataset of windows (unbatched), then mix across files
        def per_file(fn):
            ds = tf.data.TFRecordDataset(fn)
            if self.task_name is not None:
                ds = ds.filter(self._proto_filter)
            
            if self.filter_successful_trajectories:
                ds = ds.filter(self._success_filter)

            # Pass filename along with each example
            ds = ds.map(lambda x: (x, fn), num_parallel_calls=tf.data.AUTOTUNE)
            # breakpoint()
            ds = ds.map(self._decode_example_with_filename, num_parallel_calls=tf.data.AUTOTUNE)
            # breakpoint()
            ds = ds.unbatch()   # windows become elements here
            return ds

        # dataset = files.interleave(
        #     per_file,
        #     cycle_length=tf.data.AUTOTUNE,
        #     num_parallel_calls=tf.data.AUTOTUNE,
        #     deterministic=not self.is_train,
        # )
        dataset = files.flat_map(
            per_file,
            # cycle_length=tf.data.AUTOTUNE,
            # num_parallel_calls=tf.data.AUTOTUNE,
            # deterministic=not self.is_train,
        )
        
        if self.is_train:
            dataset = dataset.shuffle(512, seed=seed, reshuffle_each_iteration=True)
            dataset = dataset.repeat()

        # yields raw serialized examples
        # dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)
        
        # Filter to get single task dataset
        # if self.task_name is not None:
            # dataset = dataset.filter(self._proto_filter)

        # yields trajectories
        # dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)
        # DEBUGGING #
        # dataset = dataset.map(self._decode_example, num_parallel_calls=None)

        # cache before add_goals because add_goals introduces randomness
        # if self.cache:
        #     dataset = dataset.cache()

        # dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # if self.states_only:
        #     dataset = dataset.map(
        #         lambda x: {
        #             k: (
        #                 x[k]
        #                 if k not in ["observations", "next_observations", "goals"]
        #                 else x[k]["proprio"]
        #             )
        #             for k in x
        #         },
        #         num_parallel_calls=tf.data.AUTOTUNE,
        #     )

        # unbatch to yield individual transitions
        # dataset = dataset.unbatch()

        return dataset
    
    def _proto_filter(self, example_proto):
        if self.task_name is None:
            return tf.constant(True)

        # Define a lightweight features dict (ONLY language, nothing else)
        features = {
            "language": tf.io.FixedLenFeature([], tf.string),
        }

        parsed = tf.io.parse_single_example(example_proto, features)
        txt = parsed["language"]
        # print(parsed.keys())

        # Case-insensitive substring match
        task = tf.strings.lower(self.task_name)
        return tf.strings.regex_full_match(tf.strings.lower(txt), ".*" + task + ".*")
    
    def _success_filter(self, example_proto: tf.Tensor) -> tf.Tensor:
        """Keep only successful trajectories: last reward != second-last reward."""
        features = {
            # rewards stored via tf.io.serialize_tensor -> bytes (tf.string)
            "rewards": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        }
        parsed = tf.io.parse_single_example(example_proto, features)
        raw = parsed["rewards"]

        def _compute():
            rewards = tf.io.parse_tensor(raw, out_type=tf.float32)  # [T]
            n = tf.shape(rewards)[0]
            def _ok():
                r_last = tf.gather(rewards, n - 1)
                r_prev = tf.gather(rewards, n - 2)
                return tf.greater(tf.abs(r_last - r_prev), 1e-6)
            return tf.cond(n >= 2, _ok, lambda: tf.constant(False))

        # If rewards missing, don't filter it out.
        return tf.cond(tf.greater(tf.strings.length(raw), 0), _compute, lambda: tf.constant(True))


    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        # "next_observations/images0": tf.uint8,
        # "next_observations/state": tf.float32,
        "actions": tf.float32,
        # "terminals": tf.bool,
        # "truncates": tf.bool,
        # "episode_id": tf.int64,  # Optional: will be added dynamically if present
    }

    def _decode_example_with_filename(self, example_proto, filename):
        """Wrapper that extracts episode_id from filename and calls _decode_example."""
        # example_proto, filename = example_proto_and_filename
        
        # Extract episode_id from filename pattern: episode_<id>.tfrecord
        def extract_episode_id(fn: tf.Tensor) -> tf.Tensor:
            parts = tf.strings.split(fn, "/")
            basename = parts[-1]  # e.g. "episode_12.tfrecord" or "libero_....tfrecord"

            # Must match exactly: episode_<digits>.tfrecord
            is_match = tf.strings.regex_full_match(basename, r"episode_[0-9]+\.tfrecord")

            # FAIL if not matched (prints basename in the error)
            check = tf.debugging.Assert(
                is_match,
                [
                    "TFRecord filename must match 'episode_<id>.tfrecord' where <id> is digits. Got:",
                    basename,
                ],
            )  # :contentReference[oaicite:1]{index=1}

            # Ensure the assert runs in the tf.data graph before we proceed
            with tf.control_dependencies([check]):  # :contentReference[oaicite:2]{index=2}
                digits = tf.strings.regex_replace(
                    basename, r"^episode_([0-9]+)\.tfrecord$", r"\1"
                )
                return tf.strings.to_number(digits, out_type=tf.int64)

            # # Extract digits if matched, else fallback
            # digits = tf.strings.regex_replace(basename, r"^episode_([0-9]+)\.tfrecord$", r"\1")

            # return tf.cond(
            #     is_match,
            #     lambda: tf.strings.to_number(digits, out_type=tf.int64),
            #     lambda: tf.constant(-1, dtype=tf.int64),
            # )
        
        filename_episode_id = extract_episode_id(filename)
        
        # Call the original decode function
        return self._decode_example(example_proto, filename_episode_id)

    def _decode_example(self, example_proto, filename_episode_id=None):
        # IMPORTANT: don't create tf.constant in the signature (import-time)
        if filename_episode_id is None:
            filename_episode_id = -1  # python int
        filename_episode_id = tf.cast(filename_episode_id, tf.int64)

        # breakpoint()
        # decode the example proto according to PROTO_TYPE_SPEC
        if self.goal_relabeling_strategy is None:
            self.PROTO_TYPE_SPEC["rewards"] = tf.float32
            self.PROTO_TYPE_SPEC["masks"] = tf.float32
            self.PROTO_TYPE_SPEC["mc_returns"] = tf.float32
        if self.use_language:
            self.PROTO_TYPE_SPEC["language"] = tf.string
        if self.use_wrist_view:
            self.PROTO_TYPE_SPEC["observations/images1"] = tf.uint8
        
        # Build features dict with special handling for episode_id
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        # Try to parse episode_id as int64 if it exists
        features["episode_id"] = tf.io.FixedLenFeature([], tf.int64, default_value=-1)
        
        parsed_features = tf.io.parse_single_example(example_proto, features)
        
        # Extract episode_id separately (it's not a serialized tensor)
        stored_episode_id = stored_episode_id = tf.constant(-1, dtype=tf.int64) # parsed_features.pop("episode_id")
        
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
        
        # ### DEBUGGER SETUP FOR `_decode_example` FUNCTION ###
        # def debug_hook(rewards, masks, mc_returns, state_tf, actions_tf, image_tf):
        #     # These arguments are now standard Numpy arrays
        #     print("\n>>> HIT BREAKPOINT <<<")
        #     # print(f"Parsed tensors: {parsed_tensors.keys()}")
            
        #     # This will pause execution here. 
        #     # You can inspect variables 'rew', 'mask', etc. in your terminal.
        #     import pdb; pdb.set_trace()
            
        #     # Return dummy value to satisfy TF graph requirements
        #     return np.array(0.0, dtype=np.float32)
        
        # _ = tf.py_function(
        #         func=debug_hook, 
        #         inp=[parsed_tensors["rewards"], parsed_tensors["masks"], parsed_tensors["mc_returns"], parsed_tensors["observations/state"], parsed_tensors["actions"], parsed_tensors["observations/images0"]], 
        #         Tout=tf.float32
        # )
        # ##########################################################################################

        # Repeat prompt for each timestep (tokenizer doesn't handle batching)
        out = {}

        # LOG: `parsed_tensors` statistics #
        # breakpoint()

        # for k, v in parsed_tensors.items():
        #     tf.print("KEY:", k, "SHAPE:", tf.shape(v))
        
        # LOG: state_tf: T+1, image_tf: T+1, rest are all T in parsed_tensors #
        # LOG: Below transformation accounts for this #
        state_tf = parsed_tensors["observations/state"][:-1]
        state_tf_ns = parsed_tensors["observations/state"] #[:1:]
        # tf.print("state tf shape: ", tf.shape(state_tf))
        actions_tf = parsed_tensors["actions"]
        image_tf = [] # Do the same as `state_tf` above to drop last time step
        image_tf.append(parsed_tensors['observations/images0'][:-1]) #[:-1])
        image_tf.append(parsed_tensors["observations/images1"][:-1])
        image_tf_ns = []
        image_tf_ns.append(parsed_tensors["observations/images0"]) #[:1:])
        image_tf_ns.append(parsed_tensors['observations/images1']) #[:1:])
        # Handle the extra time step later

        ah = self.config.model.action_horizon
        T = tf.shape(state_tf)[0]

        print(f"\n\n\nT in _decode_example: {T}\n\n\n")

        # number of valid windows = T - (ah - 1)
        W = T - ah + 1
        start_idx = tf.range(W)
        # THIS SHOULD BE THE NEXT STATE FOR CHUNKING: BUG FIX #
        start_idx_ns = tf.range(ah, T + 1) # For next states/images;

        if 'rewards' in parsed_tensors:
            # Each tensor below has shape (T,); W = T - ah + 1: Maximum action chunk index #
            rewards_tf = parsed_tensors["rewards"]
            masks_tf = parsed_tensors["masks"]
            mc_returns_tf = parsed_tensors["mc_returns"]

            # rewards_tf = tf.gather(parsed_tensors["rewards"], start_idx)[: -1]
            # masks_tf = tf.gather(parsed_tensors["masks"], start_idx)[: -1]
            # mc_returns_tf = tf.gather(parsed_tensors["mc_returns"], start_idx)[: -1]
            # breakpoint()

            # Optionally override rewards: 0 at all steps, 1 at final step; build rewards, masks, and mc_returns tensors;
            if self.final_step_sparse_reward:
                # NOTE: Use this specifically for offline data for now; We assume online data already handles rewards appropriately; #
                # NOTE: Semantics of rewards in buffer #
                # rewards: (T,) that have alredy taken out last timestep of 1.0 reward and now simply decay by gamma till start;
                # masks: (T,): always seem to be 1.0 till start;
                # mc_returns: (T,): discount_cumsum(rewards)
                # To account for this, we fist add a dummy timestep to rewards, masks, and mc_returns;

                # breakpoint()
                # T_rew = tf.shape(rewards_tf)[0] # T
                # extended_rewards = tf.concat([tf.zeros([T_rew], dtype=tf.float32), [1.0]], axis=0) # (T + 1,)
                # extended_masks = tf.concat([tf.ones([T_rew], dtype=tf.float32), [0.0]], axis=0) # (T + 1,)
                # gamma = tf.constant(self.discount, dtype=tf.float32)
                # rev_rewards = tf.reverse(extended_rewards, axis=[0])
                # def scan_fn(acc, r):
                #     return r + gamma * acc
                # rev_returns = tf.scan(scan_fn, rev_rewards)
                # extended_returns = tf.reverse(rev_returns, axis=[0])


                num_steps = tf.shape(rewards_tf)[0] # (T,)
                # Start with all zeros
                rewards = tf.zeros_like(rewards_tf) # (T,)
                # Set last index to 1.0
                last_idx = num_steps - 1
                rewards = tf.tensor_scatter_nd_update(
                    rewards,
                    indices=tf.reshape(last_idx, [1, 1]),  # [[last_idx]]
                    updates=tf.constant([1.0], dtype=rewards.dtype),
                ) # Set last timestep reward to 1.0 rest 0.0
                
                rewards_tf = rewards

                # Keep masks / mc_returns as-is (or adjust later if you want them consistent)
                gamma = tf.constant(self.discount, dtype=rewards_tf.dtype)
                rev_rewards = tf.reverse(rewards_tf, axis=[0])
                
                def scan_fn(acc, r):
                    return r + gamma * acc
                
                rev_returns = tf.scan(
                    scan_fn,
                    rev_rewards,
                )

                mc_returns_tf = tf.reverse(rev_returns, axis=[0])

                # Update masks_tf
                masks = tf.ones_like(rewards_tf)
                masks = tf.tensor_scatter_nd_update(
                    masks,
                    indices=tf.reshape(last_idx, [1, 1]),
                    updates=tf.constant([0.0], dtype=masks.dtype),
                )
                masks_tf = masks
        
            # Fix code with regard to action chunking #
            # Reward at time `t` is sum of rewards for all actions in the action chunk starting at time `t` #
            # Sum rewards over the window [t : t + ah] for each starting timestep
            rewards_tf_chunked = tf.map_fn(
                lambda t: tf.reduce_sum(rewards_tf[t : t + ah]),
                start_idx,
                fn_output_signature=tf.float32,
            )
            # rewards_tf_chunked = rewards_tf_chunked[:-1]

            masks_tf_chunked = tf.map_fn(
                lambda t: tf.reduce_min(masks_tf[t : t + ah]),
                start_idx,
                fn_output_signature=tf.float32,
            )
            # masks_tf_chunked = masks_tf_chunked[:-1]

            mc_returns_tf = tf.gather(mc_returns_tf, start_idx)
            # rewards_tf: [T] step rewards
            # masks_tf:   [T] step masks in {0,1} (0 means terminal at that step)
            gamma = tf.constant(self.discount, dtype=rewards_tf.dtype)
            ah = self.config.model.action_horizon

            # ---- 1) immediate chunk reward at every start t: sum r[t:t+ah] ----
            # frames: [W, ah] where W = T - ah + 1 (pad_end=False)
            reward_frames = tf.signal.frame(
                rewards_tf, frame_length=ah, frame_step=1, pad_end=False
            )
            rewards_chunk = tf.reduce_sum(reward_frames, axis=-1)  # [W]

            # ---- 2) chunk mask at start t: min mask over the chunk ----
            mask_frames = tf.signal.frame(
                masks_tf, frame_length=ah, frame_step=1, pad_end=False
            )
            masks_chunk = tf.reduce_min(mask_frames, axis=-1)  # [W]

            # ---- 3) stride-ah discounted return:
            # R[t] = rewards_chunk[t] + gamma * masks_chunk[t] * R[t+ah]
            def stride_chunk_mc_returns(rewards_chunk, masks_chunk, gamma, ah):
                W = tf.shape(rewards_chunk)[0]
                ta = tf.TensorArray(
                    dtype=rewards_chunk.dtype,
                    size=W,
                    element_shape=tf.TensorShape([]),  # scalar each step
                    clear_after_read=False,
                )

                i0 = W - 1

                def cond(i, ta):
                    return i >= 0

                def body(i, ta):
                    next_i = i + ah
                    next_val = tf.cond(
                        next_i < W,
                        lambda: ta.read(next_i),
                        lambda: tf.zeros([], dtype=rewards_chunk.dtype),
                    )
                    val = rewards_chunk[i] + gamma * masks_chunk[i] * next_val
                    ta = ta.write(i, val)
                    return i - 1, ta

                _, ta = tf.while_loop(cond, body, [i0, ta])
                return ta.stack()  # [W]

            mc_returns_tf_chunked = stride_chunk_mc_returns(rewards_chunk, masks_chunk, gamma, ah)  # [W]


            
            out["rewards"] = rewards_tf_chunked # (W=T - ah + 1,)
            out["masks"] = masks_tf_chunked # (W=T - ah + 1,)
            # out["mc_returns"] = mc_returns_tf # (W=T - ah + 1,)
            out["mc_returns"] = mc_returns_tf_chunked # (W=T - ah + 1,)

        actions_tf = tf.map_fn(
            lambda t: actions_tf[t : t + ah],
            start_idx,
            fn_output_signature=tf.float32,
        ) # (W=T - ah + 1, ah)

        # breakpoint()

        # Get all tensors at corresponding timesteps #
        state_tf = tf.gather(state_tf, start_idx) # (W+1=T - ah, state_dim)
        image_tf[0] = tf.gather(image_tf[0], start_idx) # (W+1=T - ah, H, W, C)
        image_tf[1] = tf.gather(image_tf[1], start_idx) # (W+1=T - ah, H, W, C)
        state_tf_ns = tf.gather(state_tf_ns, start_idx_ns) # (W=T - ah, state_dim)
        image_tf_ns[0] = tf.gather(image_tf_ns[0], start_idx_ns) # (W=T - ah, H, W, C)
        image_tf_ns[1] = tf.gather(image_tf_ns[1], start_idx_ns) # (W=T - ah, H, W, C)
        length = tf.shape(actions_tf)[0] # (W=T - ah + 1,)
        prompt_tf = tf.repeat(parsed_tensors["language"][None], repeats=length)  # shape: (length,) # (W=T - ah + 1,)
        out['prompt'] = prompt_tf

        ### DEBUG ###
        # def debug_hook(state, image0, image1, prompt):
        #     print("\n>>> HIT BREAKPOINT <<<")
        #     import pdb; pdb.set_trace()
        #     return np.array(0.0, dtype=np.float32)
        # _ = tf.py_function(
        #         func=debug_hook, 
        #         inp=[state_tf, image_tf[0], image_tf[1], prompt_tf], 
        #         Tout=tf.float32
        # )
        ##########################################################################################
        
        def _apply_data_transforms_numpy(
            state, actions,
            img0, img1,
            prompt,
            # Next states/images
            state_ns, image0_ns, image1_ns,
        ):  
            if hasattr(state, "numpy"):
                state = state.numpy()
            if hasattr(actions, "numpy"):
                actions = actions.numpy()
            if hasattr(img0, "numpy"):
                img0 = img0.numpy()
            if hasattr(img1, "numpy"):
                img1 = img1.numpy()
            if hasattr(prompt, "numpy"):
                prompt = prompt.numpy()
            if hasattr(state_ns, "numpy"):
                state_ns = state_ns.numpy()
            if hasattr(image0_ns, "numpy"):
                image0_ns = image0_ns.numpy()
            if hasattr(image1_ns, "numpy"):
                image1_ns = image1_ns.numpy()

            if img0.ndim == 4:   # [T,H,W,C]
                img0 = img0[:, ::-1, ::-1, :]
                img1 = img1[:, ::-1, ::-1, :]
                image0_ns = image0_ns[:, ::-1, ::-1, :]
                image1_ns = image1_ns[:, ::-1, ::-1, :]
            else:                # [H,W,C]
                img0 = img0[::-1, ::-1, :]
                img1 = img1[::-1, ::-1, :]
                image0_ns = image0_ns[::-1, ::-1, :]
                image1_ns = image1_ns[::-1, ::-1, :]

            if self.use_8D and state.shape[-1] != 8:
                state = convert_state_15_to_8(state)
                state_ns = convert_state_15_to_8(state_ns)

            # Reconstruct EXACT input dict
            input_dict = {
                "observation/state": state,
                "actions": actions,
                "observation/image": img0,
                "observation/wrist_image": img1, 
                "prompt": prompt,            
                }
            
            # Apply OpenPI transform chain
            out = self.data_transforms(input_dict)

            # Apply to next states/images
            input_dict_ns = {
                "observation/state": state_ns,
                "observation/image": image0_ns,
                "observation/wrist_image": image1_ns,
                "prompt": prompt,
            }
            out_ns = self.data_transforms(input_dict_ns)

            if "token_ar_mask" not in out:
                out["token_ar_mask"] = np.zeros_like(out["tokenized_prompt_mask"], dtype=np.int32)

            if "token_loss_mask" not in out:
                out["token_loss_mask"] = np.ones_like(out["tokenized_prompt_mask"], dtype=np.int32)
            
            

            # Return EXACT values, in EXACT order:
            return [
                out["state"],                          # 0
                out["actions"],                        # 1

                out["image"]["base_0_rgb"],            # 2
                out["image"]["left_wrist_0_rgb"],      # 3
                out["image"]["right_wrist_0_rgb"],     # 4

                out["image_mask"]["base_0_rgb"],       # 5
                out["image_mask"]["left_wrist_0_rgb"], # 6
                out["image_mask"]["right_wrist_0_rgb"],# 7

                out["tokenized_prompt"],               # 8
                out["tokenized_prompt_mask"],          # 9
                out["token_ar_mask"],                  # 10
                out["token_loss_mask"],                # 11

                # Next states/images
                out_ns["state"],                          # 12

                out_ns["image"]["base_0_rgb"],            # 13
                out_ns["image"]["left_wrist_0_rgb"],      # 14
                out_ns["image"]["right_wrist_0_rgb"],     # 15

                out_ns["image_mask"]["base_0_rgb"],       # 16
                out_ns["image_mask"]["left_wrist_0_rgb"], # 17
                out_ns["image_mask"]["right_wrist_0_rgb"],# 18
            ]

        outputs = tf.py_function(
            func=_apply_data_transforms_numpy,
            inp=[
                state_tf, # (W+1,)
                actions_tf, # (W,)
                image_tf[0], # (W+1, H, W, C)
                image_tf[1], # (W+1, H, W, C)
                prompt_tf, # (W,)
                state_tf_ns, # (W,)
                image_tf_ns[0], # (W, H, W, C)
                image_tf_ns[1], # (W, H, W, C)
            ],
            Tout=[
                tf.float32,  # state
                tf.float32,  # actions

                tf.float32,  # base_0_rgb
                tf.float32,  # left_wrist_0_rgb
                tf.float32,  # right_wrist_0_rgb

                tf.bool,     # mask base
                tf.bool,     # mask left
                tf.bool,     # mask right

                tf.int32,    # tokenized_prompt
                tf.bool,     # tokenized_prompt_mask
                tf.int32,     # token_ar_mask
                tf.bool,     # token_loss_mask

                tf.float32,  # next state

                # Next images
                tf.float32,  # base_0_rgb
                tf.float32,  # left_wrist_0_rgb
                tf.float32,  # right_wrist_0_rgb

                # Next image masks
                tf.bool,     # mask base
                tf.bool,     # mask left
                tf.bool,     # mask right
            ]
        )
        # breakpoint()

        ### DEBUGGER SETUP FOR `_decode_example` FUNCTION ###
        # def debug_hook(states, actions, image_base, image_left, image_right, image_mask_base, image_mask_left, image_mask_right, tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask):
        #     # These arguments are now standard Numpy arrays
        #     print("\n>>> HIT BREAKPOINT <<<")
        #     # print(f"Parsed tensors: {parsed_tensors.keys()}")
            
        #     # This will pause execution here. 
        #     # You can inspect variables 'rew', 'mask', etc. in your terminal.
        #     import pdb; pdb.set_trace()
            
        #     # Return dummy value to satisfy TF graph requirements
        #     return np.array(0.0, dtype=np.float32)
        
        # _ = tf.py_function(
        #         func=debug_hook, 
        #         inp=[*outputs], 
        #         Tout=tf.float32
        # )
        ##########################################################################################

        idx = 0
        out['observations'] = {}
        out['observations']["proprio"] = outputs[idx]; idx += 1
        out["actions"] = outputs[idx]; idx += 1 #drop the last action to align dimensions

        
        out['observations']["image"] = outputs[idx]; idx += 1
        out['observations']["wrist_image"] = outputs[idx]; idx += 1
        out['observations']["image_3"] = outputs[idx]; idx += 1


        out["observations_image_mask"] = {}
        out["observations_image_mask"]["image"] = outputs[idx]; idx += 1
        out["observations_image_mask"]["wrist_image"] = outputs[idx]; idx += 1
        out["observations_image_mask"]["image_3"] = outputs[idx]; idx += 1

        out["tokenized_prompt"] = outputs[idx]; idx += 1
        out["tokenized_prompt_mask"]= outputs[idx]; idx += 1
        out["token_ar_mask"] = outputs[idx]; idx += 1
        out["token_loss_mask"] = outputs[idx]; idx += 1
        
        out['prompt'] = out['prompt']

        # Apply to next states/images
        out['next_observations'] = {}
        out['next_observations']["proprio"] = outputs[idx]; idx += 1

        out['next_observations']["image"] = outputs[idx]; idx += 1
        out['next_observations']["wrist_image"] = outputs[idx]; idx += 1
        out['next_observations']["image_3"] = outputs[idx]; idx += 1

        out['next_observations_image_mask'] = {}
        out['next_observations_image_mask']["image"] = outputs[idx]; idx += 1
        out['next_observations_image_mask']["wrist_image"] = outputs[idx]; idx += 1
        out['next_observations_image_mask']["image_3"] = outputs[idx]; idx += 1

        # Add episode ID and episode timestep
        # Priority: 1) stored_episode_id from TFRecord, 2) filename_episode_id from filename, 3) -1 as fallback
        episode_id = tf.cond(
            tf.not_equal(stored_episode_id, -1),
            lambda: stored_episode_id,
            lambda: tf.cond(
                tf.not_equal(filename_episode_id, -1),
                lambda: filename_episode_id,
                lambda: tf.constant(-1, dtype=tf.int64)
            )
        )
        # Repeat the episode ID for all windows in this trajectory
        episode_ids = tf.fill([W], tf.cast(episode_id, tf.int32))
        out['episode_id'] = episode_ids
        
        # Episode timestep is the window start index within the trajectory
        episode_timesteps = tf.cast(start_idx, tf.int32)
        out['episode_timestep'] = episode_timesteps
        
        # Add timesteps (same as episode_timestep for now, represents position within episode)
        out['timesteps'] = episode_timesteps

        # out['next_observations'] = {}
        # out['next_observations_image_mask'] = {}

        # # if 'rewards' in parsed_tensors:
        # #     out["rewards"] = rewards_tf
        # #     out["masks"] = masks_tf
        # #     out["mc_returns"] = mc_returns_tf

        # states = out['observations']["proprio"]
        # out['observations']["proprio"] = states[:-1]
        
        # out['next_observations']["proprio"] = states[1:]
        # breakpoint()
        # if self.states_only:
        #     parsed_tensors["observations/images0"] = None
        #     parsed_tensors["next_observations/images0"] = None
        # else:
        # if not self.states_only:
        #     # images = out["image"]
        #     # parsed_tensors["observations/images0"] = images[:-1]
        #     # parsed_tensors["next_observations/images0"] = images[1:]
        #     # if self.use_wrist_view:
        #     #     wrist_images = parsed_tensors["observations/images1"]
        #     #     parsed_tensors["observations/images1"] = wrist_images[:-1]
        #     #     parsed_tensors["next_observations/images1"] = wrist_images[1:]
        #     for cam in out["observations_image_mask"].keys():
        #         img = out['observations'][cam]
                
        #         out['observations'][cam] = img[:-1]
        #         out['next_observations'][cam] = img[1:]
                

        #         mask = out["observations_image_mask"][cam]
        #         out["observations_image_mask"][cam] = mask[:-1]
        #         out['next_observations_image_mask'][cam] = mask[1:]
        #         # tf.print("here 2")

        terminals = tf.zeros([W], dtype=tf.bool)
        last_idx = W - 1
        terminals = tf.tensor_scatter_nd_update(
            terminals,
            indices=tf.reshape(last_idx, [1, 1]),  # [[last_idx]]
            updates=tf.constant([True], dtype=terminals.dtype),
        ) # Set last timestep reward to 1.0 rest 0.0
        out['terminals'] = terminals
        out['truncates'] = tf.zeros([W], dtype=tf.bool)
        last_idx = W - 1
        truncates = tf.zeros([W], dtype=tf.bool)
        truncates = tf.tensor_scatter_nd_update(
            truncates,
            indices=tf.reshape(last_idx, [1, 1]),  # [[last_idx]]
            updates=tf.constant([True], dtype=truncates.dtype),
        ) # Set last timestep reward to 1.0 rest 0.0
        out['truncates'] = truncates

        def _encode_clip(text_tensor):
            """Run CLIP text encoder and return a 512-D embedding (or) use a cached embedding dict."""
            # Convert TF string to Python str
            text_str = text_tensor.numpy().decode("utf-8")
            tokens = clip.tokenize([text_str]).to(self._clip_device)
            with torch.no_grad():
                emb = self._clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            text_features = emb.cpu().numpy().squeeze().astype("float32")
            return text_features

        clip_emb = tf.py_function(
            func=_encode_clip,
            inp=[out["prompt"][0]],
            Tout=tf.float32,
        )
        clip_emb.set_shape([512])  # ViT-B/32 output dim
        num_samples = tf.shape(out["prompt"])[0]
        clip_emb = tf.repeat(clip_emb[None, :], num_samples, axis=0)
        out['observations']["language"] = clip_emb

        # Keep prompt for inference #
        MAX_PROMPT_BYTES = 256
        def encode_prompt_to_bytes(prompt_str: tf.Tensor) -> tf.Tensor:
            # prompt_str: scalar tf.string
            b = tf.io.encode_base64(prompt_str)  # or tf.strings.unicode_encode if you prefer
            b = tf.io.decode_base64(b)          # ensure raw bytes, no newline, etc.
            byte_values = tf.io.decode_raw(b, tf.uint8)  # [L]
            byte_values = byte_values[:MAX_PROMPT_BYTES]
            pad_len = MAX_PROMPT_BYTES - tf.shape(byte_values)[0]
            padded = tf.pad(byte_values, [[0, pad_len]])
            return padded  # [MAX_PROMPT_BYTES]
        
        encoded_prompts = tf.map_fn(
            encode_prompt_to_bytes,
            out["prompt"],
            fn_output_signature=tf.TensorSpec([MAX_PROMPT_BYTES], tf.uint8),
        )
        out["prompt_bytes"] = encoded_prompts
        # breakpoint()
        out.pop("prompt")

        # breakpoint()
        # print_tensor_tree("OUT", out)
        return out
    # {
    #     'image': obs_images,
    #     'next_image': next_images,
    #     'state': obs_state,
    #     'next_state': next_state,
    #     'image_mask': obs_images_mask,
    #     'next_image_mask': obs_next_images_mask,
    #     'actions': actions_window,
    #     'tokenized_prompt': tf.gather(out["tokenized_prompt"], start_idx),
    #     'tokenized_prompt_mask': tf.gather(out["tokenized_prompt_mask"], start_idx),
    #     'token_ar_mask': tf.gather(out['token_ar_mask'], start_idx),
    #     'token_loss_mask': tf.gather(out['token_loss_mask'], start_idx),
    #     'prompt': tf.gather(out['prompt'], start_idx),
    #     "terminals": terminals,
    #     "truncates": truncates,
    #      }
 
    def iterator(self, batch_size, training=True):

        return self.tf_dataset.batch(
                batch_size,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                # drop_remainder=True,
                drop_remainder=False,
                deterministic=not self.is_train,).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        # for batch in tf_iter:
        #     flat = {}
        #     flat_2 = {}

        #     # observations
        #     for k, v in batch["observations"].items():
        #         if k != "language":
        #             flat["observation/" + k] = v

        #     # # # next_observations
        #     for k, v in batch["next_observations"].items():
        #         flat_2["observation/" + k] = v

        #     # actions
        #     flat["actions"] = batch["actions"]

        #     # prompt (language)
        #     if "prompt" in batch:
        #         flat["prompt"] = batch["prompt"]
        #         flat_2["prompt"] = batch['prompt']
            
        #     # print("flat keys: ", flat.keys())

        #     output = self._apply_data_transforms(flat, training=training)
        #     # output_2 = self._apply_data_transforms(flat_2, training=training)
        #     # output['next_image'] = output_2['image']
        #     # output['next_state'] = output_2['state']
        #     # output['language'] = batch['observations']['language']
        #     yield output


    def _apply_data_transforms(self, batch, training=False):
        """
        Apply self.data_transforms (which expect single-sample dicts)
        across a batched dictionary.
        """

        B = next(iter(batch.values())).shape[0]

        # Split into per-sample dicts (zero copy views)
        # samples = []
        # for i in range(B):
        #     samples.append(
        #         jax.tree_util.tree_map(lambda x: x[i], batch)
        #     )

        # for s in samples:

        #     # ---- Fix prompt decoding ----
        #     # if "prompt" in s:
        #     #     p = s["prompt"]
        #     #     if isinstance(p, np.ndarray):
        #     #         if p.dtype.type is np.bytes_ or (p.dtype == object and isinstance(p[0], bytes)):
        #     #             s["prompt"] = np.array([x.decode("utf-8") for x in p], dtype=object)
        #     #     elif isinstance(p, bytes):
        #     #         s["prompt"] = p.decode("utf-8")
                
        #         # print("Prompt verification: ", s['prompt'])

        #     # ---- Fix proprio → state renaming ----
        #     if "observation/proprio" in s:
        #         s["observation/state"] = s.pop("observation/proprio")

        #     if "next_observation/proprio" in s:
        #         s["next_observation/state"] = s.pop("next_observation/proprio")
        #     if "actions" in s:
        #         s["actions"] = np.array(s["actions"], copy=True)
        # # Apply your pre-defined CompositeTransform
        # # print("s keys: ", samples[0].keys())
        # transformed = [self.data_transforms(s) for s in samples]
        # # print(transformed[0].keys())
        # out = self.batch_stack(transformed)


        # Split into per-sample dicts (zero copy views)
        batch['observation/state'] = batch.pop("observation/proprio")
        # print(batch.keys())
        if "actions" in batch:
            batch["actions"] = np.array(batch["actions"], copy=True)
        out = self.data_transforms(batch)
        return out
    
    def batch_stack(self, samples):
        return jax.tree_util.tree_map(
            lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
            *samples
        )



def save_trajectory_as_tfrecord(trajectory: Dict[str, np.ndarray], path: str):
    # breakpoint()
    def tensor_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
        )

    def bytes_feature(value: bytes):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value])
        )

    assert path.endswith(".tfrecord")
    assert trajectory.keys() >= {
        "observations",
        "next_observations",
        "actions",
    }
    assert type(trajectory["observations"]) == list
    assert trajectory["observations"][0].keys() >= {"image", "proprio", "prompt"}

    if tf.io.gfile.exists(path):
        print(f"Warning: Removing existing file at {path}")
        try:
            tf.io.gfile.remove(path)
        except Exception as e:
            breakpoint()

    tf.io.gfile.makedirs(os.path.dirname(path))
    try:
        language_bytes = trajectory["observations"][0]["prompt"].encode("utf-8")
        print(language_bytes)
    except:
        print("no language")
    # else:
        # language_bytes = "".encode("utf-8")
    
    # Append the last step of 'next_observations' to 'observations' for dumping to tfrecord #
    # Goes properly with the convention followed after refactor of ImageReplayBufferPi #
    trajectory["observations"].append(trajectory["next_observations"][-1])

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
                    "observations/images1": tensor_feature(
                        np.array(
                            [o["wrist_image"] for o in trajectory["observations"]], 
                            dtype=np.uint8)
                    ),
                    "observations/state": tensor_feature(
                        np.array(
                            [o["proprio"] for o in trajectory["observations"]],
                            dtype=np.float32,
                        )
                    ),
                    # "next_observations/images0": tensor_feature(
                    #     np.array(
                    #         [o["image"] for o in trajectory["next_observations"]],
                    #         dtype=np.uint8,
                    #     )
                    # ),
                    # "next_observations/state": tensor_feature(
                    #     np.array(
                    #         [o["proprio"] for o in trajectory["next_observations"]],
                    #         dtype=np.float32,
                    #     )
                    # ),
                    "actions": tensor_feature(
                        np.array(trajectory["actions"], dtype=np.float32) # DO NOT drop last step
                    ),
                    **(
                        {
                            "rewards": tensor_feature(
                                np.array(trajectory["rewards"], dtype=np.float32) # DO NOT drop last step
                            ),
                            "masks": tensor_feature(
                                np.array(trajectory["masks"], dtype=np.float32) # DO NOT drop last step
                            ),
                            "mc_returns": tensor_feature(
                                np.array(trajectory["mc_returns"], dtype=np.float32) # DO NOT drop last step
                            ),
                            "terminals": tensor_feature(
                                np.array(trajectory["terminals"], dtype=np.float32) # DO NOT drop last step
                            ),
                            "truncates": tensor_feature(
                                np.array(trajectory["truncates"], dtype=np.float32) # DO NOT drop last step
                            ),
                        }
                        if "rewards" in trajectory
                        else {}
                    ),
                    "language":  bytes_feature(language_bytes),
                }
            )
        )
        writer.write(example.SerializeToString())

import numpy as np
from robosuite.utils.transform_utils import quat2axisangle

def convert_state_15_to_8(state_15: np.ndarray):
    """
    Convert your 15-D state vector into the 8-D OpenPI format:
        [eef_pos (3), axis_angle (3), gripper (1), dummy (1)]
    NOTE:
        Uses quat2axisangle from robosuite.
        Works for both batched (B, 15) and unbatched (15,) inputs.
    """

    # Ensure batch
    state_15 = np.asarray(state_15)
    batched = state_15.ndim == 2
    if not batched:
        state_15 = state_15[None, :]   # → (1, 15)

    B = state_15.shape[0]

    # Extract eef pos and quat
    eef_pos = state_15[:, 0:3]            # (B, 3)
    quat    = state_15[:, 3:7]            # (B, 4)

    # Convert quaternion → axis-angle using robosuite
    axis_angle = np.zeros((B, 3), dtype=np.float32)
    for i in range(B):
        axis_angle[i] = quat2axisangle(quat[i])

    # Extract gripper (last element)
    gripper = state_15[:, 14:15]          # (B, 1)

    # Build final 8-D vector
    # [pos(3), axis(3), gripper(1), filler(1)]
    out = np.concatenate(
        [eef_pos, axis_angle, gripper, np.zeros((B,1), dtype=np.float32)],
        axis=1
    )   # shape (B, 8)

    return out if batched else out[0]

def print_tensor_tree(prefix, obj):
    """
    Recursively print shapes and dtypes of all Tensors in a nested structure.
    Handles dicts, lists, tuples, and tensors.
    Uses tf.print so it works inside tf.data / tf.function.
    """

    # Case 1: Tensor -> print its shape + dtype
    if isinstance(obj, tf.Tensor):
        tf.print(prefix, "SHAPE:", tf.shape(obj), "DTYPE:", obj.dtype)
        return

    # Case 2: dict -> recurse on each key
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = prefix + "/" + str(k)
            print_tensor_tree(child_prefix, v)
        return

    # Case 3: list or tuple -> recurse on each element
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            child_prefix = prefix + f"[{i}]"
            print_tensor_tree(child_prefix, v)
        return

    # Case 4: anything else (scalar, None, etc.) -> just print type
    tf.print(prefix, "NON-TENSOR LEAF OF TYPE:", str(type(obj)))

if __name__ == "__main__":
    import os
    import glob
    import numpy as np

    tf.config.set_visible_devices([], 'GPU')
    inspect_tfrecords()

    # -------------------------------
    # Load config
    # -------------------------------
    config = get_config("pi05_libero_custom_low_mem")
    print("Assets dirs:", config.assets_dirs)
    print("Model config:", config.model)

    data_config = config.data.create(config.assets_dirs, config.model)
    print("data config:", data_config)

    # -------------------------------
    # Locate TFRecords
    # -------------------------------
    # tfrecord_dir = "/data/hf_cache/datasets/LIBERO/libero_10_tf"
    # tfrecord_dir = "/home/sreyasv/Projects/PolicyAgnosticRL/results/image_replay_buffer"

    # data_paths = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
    data_paths = "/data/hf_cache/datasets/LIBERO/libero_10_tf/*.tfrecord"

    # if not data_paths:
    #     raise FileNotFoundError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"[INFO] Found {len(data_paths)} TFRecord files.")
    print("[INFO] Using first 5 files for integration test.\n")

    # -------------------------------
    # Construct Replay Buffer
    # -------------------------------
    buffer = ImageReplayBufferPi(
        data_paths=data_paths,
        seed=42,
        use_language=True,
        cache=False,
        tfrecords_include_next_observations=False,
        config=config,
        # task_name= "put both moka pots on the stove" #"put the yellow and white mug in the microwave and close it",
    )

    iterator = buffer.iterator(batch_size=64)

    print("[INFO] Fetching one batch...\n")

    # -------------------------------
    # Your iterator returns:
    #   (Observation, actions)
    # -------------------------------
    output = next(iterator)

    # breakpoint()

    print("\n========== BATCH STRUCTURE ==========\n")
    print("output keys: ", output.keys())
    print("prompt: ", output['prompt'])

    # -------------------------
    # Actions
    # -------------------------
    print("------- ACTIONS -------")
    print("Actions shape:", output['actions'].shape)
    print("First action:", output['actions'][0], "\n")

    # -------------------------
    # Images
    # # -------------------------
    # print("------- IMAGES -------")
    # print("Image cameras:", list(output['image'].keys()))
    # # print("Next image cameras:", list(output['next_image'].keys()), "\n")

    # for cam, img in output['image'].items():
    #     print(f"[{cam}] image shape:", img.shape)
    #     print(f"First pixel: {img[0,0,0]}")
    #     break
    # # first_img = output["image"]["base_0_rgb"][0]    # [224,224,3]

    # # save_image_tensor_as_png(first_img, "first_base_rgb.png")
    # # -------------------------
    # # State
    # # -------------------------
    print("\n------- STATE -------")
    print("State shape:", output['observations']['proprio'].shape)
    print("State[0]:", output['observations']['proprio'][0])

    # Next state
    # print("\nNext state shape:", output['next_state'].shape)
    # print("Next state[0]:", output['next_state'][0])

    # -------------------------
    # Tokenized Prompt
    # -------------------------
    print("\n------- TOKENIZED PROMPT -------")

    tp = output.get("tokenized_prompt", None)
    tpm = output.get("tokenized_prompt_mask", None)

    if tp is None:
        print("No tokenized prompt.")
    else:
        print("Tokenized prompt shape:", tp.shape)
        print("Prompt tokens [0,:10]:", tp[0, :10])

    if tpm is not None:
        print("\nPrompt mask shape:", tpm.shape)
        print("Mask[0,:10]:", tpm[0, :10])

    # -------------------------
    # Language string
    # -------------------------
    print("\n------- LANGUAGE (raw) -------")
    if output.get("language", None) is not None:
        print("Language example:", output["language"][0].shape)
    else:
        print("No raw language field.")

    # -------------------------
    # Episode ID and Timestep
    # -------------------------
    print("\n------- EPISODE INFO -------")
    if output.get("episode_id", None) is not None:
        print("Episode ID shape:", output["episode_id"].shape)
        print("Episode ID (first 3):", output["episode_id"][:3])
        print("Episode ID unique values:", np.unique(output["episode_id"]))
    else:
        print("No episode_id field.")
    
    if output.get("episode_timestep", None) is not None:
        print("\nEpisode timestep shape:", output["episode_timestep"].shape)
        print("Episode timestep (first 10):", output["episode_timestep"][:10])
    else:
        print("No episode_timestep field.")
    
    if output.get("timesteps", None) is not None:
        print("\nTimesteps shape:", output["timesteps"].shape)
        print("Timesteps (first 10):", output["timesteps"][:10])
    else:
        print("No timesteps field.")

    print("\n==========================================")
    print("   ✅ PI-0.5 ImageReplayBuffer test PASSED")
    print("==========================================\n") 

    import time

    num_batches = 20
    start = time.time()

    for _ in range(num_batches):
        batch = next(iterator)

    end = time.time()

    avg = (end - start) / num_batches

    print(f"\n⏱ Average time per batch over {num_batches} batches: {avg:.4f} seconds")
    print(f"Total time: {end - start:.4f} seconds\n")
