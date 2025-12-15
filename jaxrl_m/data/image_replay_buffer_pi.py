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
from PIL import Image

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
        self.data_transforms = _transforms.compose(self.data_transforms)
        self.task_name = task_name
        self.use_8D = use_8D
        self.final_step_sparse_reward = final_step_sparse_reward
        self.discount = discount
        
        dataset = self._construct_tf_dataset(data_paths, seed)

        if train:
            dataset = dataset.repeat()
            if self.task_name is None:
                dataset = dataset.shuffle(512, reshuffle_each_iteration=True)


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
        
        # Filter to get single task dataset
        if self.task_name is not None:
            dataset = dataset.filter(self._proto_filter)

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
        if self.use_language:
            self.PROTO_TYPE_SPEC["language"] = tf.string
        if self.use_wrist_view:
            self.PROTO_TYPE_SPEC["observations/images1"] = tf.uint8
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
        # Repeat prompt for each timestep (tokenizer doesn't handle batching)
        out = {}

        # LOG: `parsed_tensors` statistics #
        # breakpoint()

        # for k, v in parsed_tensors.items():
        #     tf.print("KEY:", k, "SHAPE:", tf.shape(v))
        
        state_tf = parsed_tensors["observations/state"][:-1] #drop the last state to align dimensions
        # tf.print("state tf shape: ", tf.shape(state_tf))
        actions_tf = parsed_tensors["actions"]
        image_tf = [parsed_tensors["observations/images0"][:-1], parsed_tensors['observations/images1'][:-1]]

        # breakpoint()
        
        
        ah = self.config.model.action_horizon
        T = tf.shape(state_tf)[0]

        # number of valid windows = T - (ah - 1)
        W = T - ah + 1
        start_idx = tf.range(W)  
        # if 'rewards' in parsed_tensors:
        #     rewards_tf = tf.gather(parsed_tensors["rewards"], start_idx)[: -1]
        #     masks_tf = tf.gather(parsed_tensors["masks"], start_idx)[: -1]
        #     mc_returns_tf = tf.gather(parsed_tensors["mc_returns"], start_idx)[: -1]
        # breakpoint()
        if 'rewards' in parsed_tensors:
            rewards_tf = tf.gather(parsed_tensors["rewards"], start_idx)[: -1]
            masks_tf = tf.gather(parsed_tensors["masks"], start_idx)[: -1]
            mc_returns_tf = tf.gather(parsed_tensors["mc_returns"], start_idx)[: -1]
            # breakpoint()

            # Optionally override rewards: 0 at all steps, 1 at final step
            if self.final_step_sparse_reward:
                # breakpoint()
                # rewards_tf has shape [W-1]; we use its shape to build the new vector
                num_steps = tf.shape(rewards_tf)[0]
                # Start with all zeros
                rewards = tf.zeros_like(rewards_tf)
                # Set last index to 1.0
                last_idx = num_steps - 1
                rewards = tf.tensor_scatter_nd_update(
                    rewards,
                    indices=tf.reshape(last_idx, [1, 1]),  # [[last_idx]]
                    updates=tf.constant([1.0], dtype=rewards.dtype),
                )
                out["rewards"] = rewards
                rewards_tf = rewards
                # Keep masks / mc_returns as-is (or adjust later if you want them consistent)
                out["masks"] = masks_tf
                # out["mc_returns"] = mc_returns_tf

                gamma = tf.constant(self.discount, dtype=rewards_tf.dtype)
                rev_rewards = tf.reverse(rewards_tf, axis=[0])
                
                def scan_fn(acc, r):
                    return r + gamma * acc
                
                rev_returns = tf.scan(
                    scan_fn,
                    rev_rewards,
                )

                mc_returns_tf = tf.reverse(rev_returns, axis=[0])
            else:
                out["rewards"] = rewards_tf
                out["masks"] = masks_tf
                out["mc_returns"] = mc_returns_tf

        actions_tf = tf.map_fn(
            lambda t: actions_tf[t : t + ah],
            start_idx,
            fn_output_signature=tf.float32,
        )

        # breakpoint()

        state_tf = tf.gather(state_tf, start_idx)
        image_tf[0] = tf.gather(image_tf[0], start_idx)
        image_tf[1] = tf.gather(image_tf[1], start_idx)
        length = tf.shape(state_tf)[0]
        prompt_tf = tf.repeat(parsed_tensors["language"][None], repeats=length)  # shape: (length,)
        # tf.print(prompt_tf)
        #  # ===== DEBUG SHAPE PRINTS =====
        # tf.print("----- DEBUG SHAPES -----")
        # tf.print("state_tf shape:", tf.shape(state_tf))
        # tf.print("actions_tf shape:", tf.shape(actions_tf))
        # tf.print("image_tf[0] shape:", tf.shape(image_tf[0]))   # base camera
        # tf.print("image_tf[1] shape:", tf.shape(image_tf[1]))   # wrist camera
        # tf.print("prompt_tf shape:", tf.shape(prompt_tf))
        # tf.print("start_idx shape:", tf.shape(start_idx))
        # tf.print("length:", length)
        # tf.print("------------------------")
        out['prompt'] = prompt_tf

        # breakpoint()

        
        def _apply_data_transforms_numpy(
            state, actions,
            img0, img1,
            prompt,
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

            if img0.ndim == 4:   # [T,H,W,C]
                img0 = img0[:, ::-1, ::-1, :]
                img1 = img1[:, ::-1, ::-1, :]
            else:                # [H,W,C]
                img0 = img0[::-1, ::-1, :]
                img1 = img1[::-1, ::-1, :]

            if self.use_8D and state.shape[-1] != 8:
                state = convert_state_15_to_8(state)

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
            ]

        outputs = tf.py_function(
            func=_apply_data_transforms_numpy,
            inp=[
                state_tf,
                actions_tf,
                image_tf[0],
                image_tf[1],
                prompt_tf,
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
            ]
        )
        # breakpoint()

        idx = 0
        out['observations'] = {}
        out['observations']["proprio"] = outputs[idx]; idx += 1
        # tf.print("state: ", tf.shape(out['state']))
        out["actions"] = outputs[idx][:-1]; idx += 1 #drop the last action to align dimensions
        # tf.print("actions: ", tf.shape(out['actions']))

        
        out['observations']["image"] = outputs[idx]; idx += 1
        # tf.print("image:1:", tf.shape(out["image"]["base_0_rgb"]))
        out['observations']["wrist_image"] = outputs[idx]; idx += 1
        # tf.print("image: 2:" ,tf.shape(out["image"]["left_wrist_0_rgb"]))
        out['observations']["image_3"] = outputs[idx]; idx += 1
        # tf.print("image:3:", tf.shape(out["image"]["right_wrist_0_rgb"]))


        out["observations_image_mask"] = {}
        out["observations_image_mask"]["image"] = outputs[idx]; idx += 1
        # tf.print("im mask :1:",out["image_mask"]["base_0_rgb"])
        out["observations_image_mask"]["wrist_image"] = outputs[idx]; idx += 1
        # tf.print(tf.shape(out["image_mask"]["left_wrist_0_rgb"]))
        out["observations_image_mask"]["image_3"] = outputs[idx]; idx += 1
        # tf.print(tf.shape(out["image_mask"]["right_wrist_0_rgb"]))

        out["tokenized_prompt"] = outputs[idx][:-1]; idx += 1
        out["tokenized_prompt_mask"]= outputs[idx][:-1]; idx += 1
        out["token_ar_mask"] = outputs[idx][:-1]; idx += 1
        out["token_loss_mask"] = outputs[idx][:-1]; idx += 1
        
        out['prompt'] = out['prompt'][:-1]

        out['next_observations'] = {}
        out['next_observations_image_mask'] = {}
        # tf.print("========== TRANSFORM OUTPUT SHAPES ==========\n",

        #     # ---- STATE ----
        #     "state shape:", tf.shape(out["state"]), "\n",
        #     "actions shape:", tf.shape(out["actions"]), "\n",

        #     # ---- IMAGES ----
        #     "image/base_0_rgb shape:", tf.shape(out["image"]["base_0_rgb"]), "\n",
        #     "image/left_wrist_0_rgb shape:", tf.shape(out["image"]["left_wrist_0_rgb"]), "\n",
        #     "image/right_wrist_0_rgb shape:", tf.shape(out["image"]["right_wrist_0_rgb"]), "\n",

        #     # ---- IMAGE MASKS ----
        #     "image_mask/base_0_rgb shape:", tf.shape(out["image_mask"]["base_0_rgb"]), "\n",
        #     "image_mask/left_wrist_0_rgb shape:", tf.shape(out["image_mask"]["left_wrist_0_rgb"]), "\n",
        #     "image_mask/right_wrist_0_rgb shape:", tf.shape(out["image_mask"]["right_wrist_0_rgb"]), "\n",

        #     # ---- TOKENS ----
        #     "tokenized_prompt shape:", tf.shape(out["tokenized_prompt"]), "\n",
        #     "tokenized_prompt_mask shape:", tf.shape(out["tokenized_prompt_mask"]), "\n",
        #     "token_ar_mask shape:", tf.shape(out["token_ar_mask"]), "\n",
        #     "token_loss_mask shape:", tf.shape(out["token_loss_mask"]), "\n",

        #     # ---- PROMPT ----
        #     "prompt shape:", tf.shape(out["prompt"]), "\n",

        #     # Footer
        #     "============================================"
        # )
        if 'rewards' in parsed_tensors:
            out["rewards"] = rewards_tf
            out["masks"] = masks_tf
            out["mc_returns"] = mc_returns_tf

        states = out['observations']["proprio"]
        out['observations']["proprio"] = states[:-1]
        
        out['next_observations']["proprio"] = states[1:]
        # breakpoint()
        # if self.states_only:
        #     parsed_tensors["observations/images0"] = None
        #     parsed_tensors["next_observations/images0"] = None
        # else:
        if not self.states_only:
            # images = out["image"]
            # parsed_tensors["observations/images0"] = images[:-1]
            # parsed_tensors["next_observations/images0"] = images[1:]
            # if self.use_wrist_view:
            #     wrist_images = parsed_tensors["observations/images1"]
            #     parsed_tensors["observations/images1"] = wrist_images[:-1]
            #     parsed_tensors["next_observations/images1"] = wrist_images[1:]
            for cam in out["observations_image_mask"].keys():
                img = out['observations'][cam]
                
                out['observations'][cam] = img[:-1]
                out['next_observations'][cam] = img[1:]
                

                mask = out["observations_image_mask"][cam]
                out["observations_image_mask"][cam] = mask[:-1]
                out['next_observations_image_mask'][cam] = mask[1:]
                # tf.print("here 2")

        out['terminals'] = tf.zeros([W-1], dtype=tf.bool)
        # # terminals[-1] = True
        out['truncates'] = tf.zeros([W-1], dtype=tf.bool)
        # truncates[-1] = True
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
                drop_remainder=True,
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
                        np.array(trajectory["actions"][:-1], dtype=np.float32)
                    ),
                    **(
                        {
                            "rewards": tensor_feature(
                                np.array(trajectory["rewards"][:-1], dtype=np.float32)
                            ),
                            "masks": tensor_feature(
                                np.array(trajectory["masks"][:-1], dtype=np.float32)
                            ),
                            "mc_returns": tensor_feature(
                                np.array(trajectory["mc_returns"][:-1], dtype=np.float32)
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
    tfrecord_dir = "/home/sreyasv/Projects/PolicyAgnosticRL/results/image_replay_buffer"
    data_paths = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")))

    if not data_paths:
        raise FileNotFoundError(f"No TFRecord files found in {tfrecord_dir}")

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
