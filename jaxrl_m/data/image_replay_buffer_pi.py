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
        use_8D=False,
        final_step_sparse_reward: bool = True, #Assume the trajectory is successful by default
        discount: float = 0.99,
        traj_sampling: bool = False,
        final_step_reward: int = 10,
        anti_sparse: bool = True, # If true, gives negative reward for every step other than final step
        filter_success: bool = False,
        num_of_traj: int = None,
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
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        if self.final_step_sparse_reward:
            logging.info("Using final step sparse reward setting. By defualt the trajectory is assumed to be successful.")
        self.final_step_reward = final_step_reward
        self.anti_sparse = anti_sparse

        self.discount = discount
        self.is_train = train
        self.traj_sampling = traj_sampling
        self.filter_success = filter_success
     
        self.num_trajectories = num_of_traj

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
        if (self.task_name is not None) or (self.filter_success):
            dataset = dataset.filter(self._proto_filter)
        
        if self.num_trajectories is not None:
            dataset = dataset.take(self.num_trajectories)

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
        if not self.traj_sampling:
            dataset = dataset.unbatch()

        return dataset
    
    def _proto_filter(self, example_proto):
        # Parse only what we need
        features = {
            "language": tf.io.FixedLenFeature([], tf.string),
            "rewards": tf.io.FixedLenFeature([], tf.string),
        }

        parsed = tf.io.parse_single_example(example_proto, features)

        # ------------------
        # Language filter
        # ------------------
        if self.task_name is None:
            language_ok = tf.constant(True)
        else:
            txt = parsed["language"]
            task = tf.strings.lower(self.task_name)
            language_ok = tf.strings.regex_full_match(
                tf.strings.lower(txt),
                ".*" + task + ".*"
            )

        # ------------------
        # Success filter
        # ------------------
        if not self.filter_success:
            return language_ok

        rewards = tf.io.parse_tensor(parsed["rewards"], tf.float32)
        final_env_reward = rewards[-1]

        # success if final reward == 1.0
        success = tf.equal(final_env_reward, 1.0)

        return tf.logical_and(language_ok, success)


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

        # for k, v in parsed_tensors.items():
        #     tf.print("KEY:", k, "SHAPE:", tf.shape(v))
        
        state_tf = parsed_tensors["observations/state"][:-1] #drop the last state to align dimensions
        state_tf_ns = parsed_tensors["observations/state"]
        # tf.print("state tf shape: ", tf.shape(state_tf))
        actions_tf = parsed_tensors["actions"]
        image_tf = [parsed_tensors["observations/images0"][:-1], parsed_tensors['observations/images1'][:-1]]
        image_tf_ns = [parsed_tensors["observations/images0"], parsed_tensors['observations/images1']]
        
        
        ah = self.config.model.action_horizon
        T = tf.shape(state_tf)[0]

        # number of valid windows = T - (ah - 1)
        W = T - ah + 1
        start_idx = tf.range(W)
        start_idx_rewards = tf.range(W)
        start_idx_ns = start_idx + ah
        if self.anti_sparse:
            rewards_raw = -tf.ones([T], dtype=tf.float32)
        else:
            rewards_raw = tf.zeros([T], dtype=tf.float32)
        dones_raw   = tf.zeros([T], dtype=tf.float32)

        if self.final_step_sparse_reward:
            success = tf.constant(True)
        else:
            final_env_reward = parsed_tensors["rewards"][-1]
            # (your logic) success if final reward == 0.0
            # success = tf.equal(final_env_reward, 0.0)
            success = tf.equal(final_env_reward, 1.0)

        def success_case_raw():
            r = tf.tensor_scatter_nd_update(
                rewards_raw, indices=tf.reshape(T - 1, [1, 1]), updates=tf.constant([self.final_step_reward], tf.float32)
            )
            d = tf.tensor_scatter_nd_update(
                dones_raw, indices=tf.reshape(T - 1, [1, 1]), updates=tf.constant([1.0], tf.float32)
            )
            return r, d

        rewards_raw, dones_raw = tf.cond(success, success_case_raw, lambda: (rewards_raw, dones_raw))
        masks_raw = 1.0 - dones_raw

        gamma = tf.constant(self.discount, dtype=tf.float32)
        gamma_vec = tf.pow(gamma, tf.range(ah, dtype=tf.float32))
        # Window/chunk them: length W
        rewards_chunked = tf.map_fn(
            lambda t: tf.reduce_sum(
                rewards_raw[t : t + ah] * gamma_vec
            ),
            start_idx_rewards,
            fn_output_signature=tf.float32,
        )

        masks_chunked = tf.map_fn(
            lambda t: tf.reduce_min(masks_raw[t : t + ah]),
            start_idx_rewards,
            fn_output_signature=tf.float32,
        )

        # Monte-Carlo returns over per-transition rewards (length L)
        rev_rewards = tf.reverse(rewards_raw, axis=[0])
        rev_returns = tf.scan(lambda acc, r: r + gamma * acc, rev_rewards)
        mc_returns  = tf.reverse(rev_returns, axis=[0])
        mc_returns = tf.gather(mc_returns, start_idx_rewards) #[:-1]  # align to L

        out["rewards"] = rewards_chunked
        out["masks"] = masks_chunked
        out["mc_returns"] = mc_returns

        base_actions_tf = actions_tf
        start_idx_next_actions = tf.range(ah, T+1)
        action_dim = tf.shape(base_actions_tf)[-1]
        pad = tf.zeros([ah, action_dim], dtype=base_actions_tf.dtype)
        padded_actions_tf = tf.concat([base_actions_tf, pad], axis=0)

        next_actions_tf = tf.map_fn(
            lambda t: padded_actions_tf[t : t + ah],
            start_idx_next_actions,
            fn_output_signature=tf.float32,
        ) # (W=T - ah, ah)


        actions_tf = tf.map_fn(
            lambda t: actions_tf[t : t + ah],
            start_idx,
            fn_output_signature=tf.float32,
        )

        state_tf = tf.gather(state_tf, start_idx)
        image_tf[0] = tf.gather(image_tf[0], start_idx)
        image_tf[1] = tf.gather(image_tf[1], start_idx)

        state_tf_ns = tf.gather(state_tf_ns, start_idx_ns) # (W=T - ah, state_dim)
        image_tf_ns[0] = tf.gather(image_tf_ns[0], start_idx_ns) # (W=T - ah, H, W, C)
        image_tf_ns[1] = tf.gather(image_tf_ns[1], start_idx_ns) # (W=T - ah, H, W, C)

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

            # if img0.ndim == 4:   # [T,H,W,C]
            #     img0 = img0[:, ::-1, ::-1, :]
            #     img1 = img1[:, ::-1, ::-1, :]
            # else:                # [H,W,C]
            #     img0 = img0[::-1, ::-1, :]
            #     img1 = img1[::-1, ::-1, :]

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
                out["token_loss_mask"] = np.ones_like(out["tokenized_prompt_mask"], dtype=np.bool_)
            
            

            # Return EXACT values, in EXACT order:
            return [
                out["state"],                          # 0
                out["actions"],                        # 1

                out["image"]["base_0_rgb"].astype(np.uint8),            # 2
                out["image"]["left_wrist_0_rgb"].astype(np.uint8),      # 3
                out["image"]["right_wrist_0_rgb"].astype(np.uint8),     # 4

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

                tf.uint8 ,  #tf.float32,  # base_0_rgb
                tf.uint8 ,  #tf.float32,  # left_wrist_0_rgb
                tf.uint8 ,  #tf.float32,  # right_wrist_0_rgb

                tf.bool,     # mask base
                tf.bool,     # mask left
                tf.bool,     # mask right

                tf.int32,    # tokenized_prompt
                tf.bool,     # tokenized_prompt_mask
                tf.int32,     # token_ar_mask
                tf.bool,     # token_loss_mask
            ]
        )

        outputs_ns = tf.py_function(
            func=_apply_data_transforms_numpy,
            inp=[
                state_tf_ns,
                next_actions_tf,
                image_tf_ns[0],
                image_tf_ns[1],
                prompt_tf,
            ],
            Tout=[
                tf.float32,  # state
                tf.float32,  # actions

                tf.uint8 ,  # base_0_rgb
                tf.uint8 ,  # left_wrist_0_rgb
                tf.uint8 ,  # right_wrist_0_rgb

                tf.bool,     # mask base
                tf.bool,     # mask left
                tf.bool,     # mask right

                tf.int32,    # tokenized_prompt
                tf.bool,     # tokenized_prompt_mask
                tf.int32,     # token_ar_mask
                tf.bool,     # token_loss_mask
            ]
        )
        

        idx = 0
        out['observations'] = {}
        out['observations']["proprio"] = outputs[idx]; idx += 1
        # tf.print("state: ", tf.shape(out['state']))
        out["actions"] = outputs[idx]; idx += 1 #drop the last action to align dimensions
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

        out["tokenized_prompt"] = outputs[idx]; idx += 1
        out["tokenized_prompt_mask"]= outputs[idx]; idx += 1
        out["token_ar_mask"] = outputs[idx]; idx += 1
        out["token_loss_mask"] = outputs[idx]; idx += 1
        
        # out['prompt'] = out['prompt']
        idx = 0
        out['next_observations'] = {}
        out['next_observations']["proprio"] = outputs_ns[idx]; idx += 1
        out['next_observations']['next_on_policy_actions'] = outputs_ns[idx]; idx += 1
        out['next_observations']["image"] = outputs_ns[idx]; idx += 1
        out['next_observations']["wrist_image"] = outputs_ns[idx]; idx += 1
        out['next_observations']["image_3"] = outputs_ns[idx]; idx += 1

        out["next_observations_image_mask"] = {}
        out["next_observations_image_mask"]["image"] = outputs_ns[idx]; idx += 1
        out["next_observations_image_mask"]["wrist_image"] = outputs_ns[idx]; idx += 1
        out["next_observations_image_mask"]["image_3"] = outputs_ns[idx]; idx += 1


        # out['next_observations'] = {}
        # out['next_observations_image_mask'] = {}
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
        # if 'rewards' in parsed_tensors:
        #     out["rewards"] = rewards_tf
        #     out["masks"] = masks_tf
        #     out["mc_returns"] = mc_returns_tf

        # states = out['observations']["proprio"]
        # out['observations']["proprio"] = states[:-1]
        
        # out['next_observations']["proprio"] = states[1:]
        # # if self.states_only:
        # #     parsed_tensors["observations/images0"] = None
        # #     parsed_tensors["next_observations/images0"] = None
        # # else:
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
        out.pop("prompt")
        # print_tensor_tree("OUT", out)
        # tf.print(
        #     "\n===== DEBUG BATCH SHAPES =====",
        #     "\nproprio:", tf.shape(out["observations"]["proprio"]),
        #     "\nactions:", tf.shape(out["actions"]),
        #     "\nrewards:", tf.shape(out.get("rewards", tf.constant([-1]))),
        #     "\nmasks:", tf.shape(out.get("masks", tf.constant([-1]))),
        #     "\nmc_returns:", tf.shape(out.get("mc_returns", tf.constant([-1]))),
        #     "\nimage:", tf.shape(out["observations"]["image"]),
        #     "\nprompt:", tf.shape(out["tokenized_prompt"]),
        #     "\n==============================",
        #     summarize=-1,
        # )
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

import os
import json
import numpy as np
import imageio
from PIL import Image

def _to_numpy(x):
    """Convert torch or numpy to numpy."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _to_uint8_frames(frames):
    """
    frames: (T,H,W,3) as float or uint8
    Robustly convert to uint8 without blowing everything to white.
    """
    frames = np.asarray(frames)
    if frames.dtype == np.uint8:
        return frames

    fmin = frames.min()
    fmax = frames.max()
    eps = 1e-6

    # Case 1: looks like [0, 1]
    if fmax <= 1.0 + 1e-3 and fmin >= -1e-3:
        frames_uint8 = np.clip(frames, 0.0, 1.0) * 255.0
        return frames_uint8.astype(np.uint8)

    # Case 2: looks like [0, 255] already (float32)
    if fmax <= 255.0 + 1.0 and fmin >= -1.0:
        frames_uint8 = np.clip(frames, 0.0, 255.0)
        return frames_uint8.astype(np.uint8)

    # Fallback: normalize by global max
    frames_norm = (frames - fmin) / (fmax - fmin + eps)
    frames_uint8 = np.clip(frames_norm * 255.0, 0.0, 255.0)
    return frames_uint8.astype(np.uint8)


def export_batch_to_json(
    output,
    task,
    texts,
    episode_id="1",
    out_dir="export",
    video_name="rgb.mp4",
    json_name="episode.json",
    max_timesteps=32,
    fps=10,
):
    """
    Export a batch into:
      - JSON (task, texts, videos, action, state, continuous_gripper_state)
      - MP4 video
      - First-frame PNG

    Conventions you specified:
      - actions: shape (1, T, 10, 32) → use [0, i, 0, :7] for i in [0, 31]
      - state:   output['observations']['proprio'] with shape (1, T, D)
                 → use [0, i, :8]
      - continuous_gripper_state[i] = state[i][-1]
      - images:  output['observations']['image'] with shape (1, T, H, W, 3)
    """
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # Extract tensors and convert to numpy
    # -----------------------
    actions_tensor = _to_numpy(output["actions"])                  # (1, T, 10, 32)
    proprio_tensor = _to_numpy(output["observations"]["proprio"])  # (1, T, D)
    images_tensor  = _to_numpy(output["observations"]["image"])    # (1, T, H, W, 3)

    T = min(max_timesteps, actions_tensor.shape[1])

    actions_list = []
    state_list = []
    gripper_list = []

    for i in range(T):
        # ----- Action: [0, i, 0, :7]
        a = actions_tensor[0, i, 0, :7].tolist()
        actions_list.append(a)

        # ----- State: [0, i, :8]
        s = proprio_tensor[0, i, :8].tolist()
        state_list.append(s)

        # ----- Gripper = last element of that state vector
        gripper_list.append(float(s[-1]))

    # -----------------------
    # Prepare frames for saving
    # -----------------------
    frames = images_tensor[0, :T]  # (T, H, W, 3)
    frames_uint8 = _to_uint8_frames(frames)

    # -----------------------
    # Save MP4 video
    # -----------------------
    video_path = os.path.join(out_dir, video_name)
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in frames_uint8:
        writer.append_data(frame)
    writer.close()

    # -----------------------
    # Save first frame PNG
    # -----------------------
    first_image_path = os.path.join(out_dir, "frame_0000.png")
    Image.fromarray(frames_uint8[0]).save(first_image_path)

    # -----------------------
    # Build JSON object
    # -----------------------
    json_dict = {
        "task": task,
        "texts": texts,  # list of strings
        "videos": [
            {"video_path": video_path}
        ],
        "latent_videos": [],
        "action": actions_list,
        "state": state_list,
        "continuous_gripper_state": gripper_list,
        "episode_id": str(episode_id),
    }

    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=4)

    print(f"[✓] Saved JSON → {json_path}")
    print(f"[✓] Saved MP4  → {video_path}")
    print(f"[✓] Saved first frame → {first_image_path}")

    return json_dict, json_path, video_path, first_image_path


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
        task_name="put both moka pots on the stove", #"put both moka pots on the stove" #"put the yellow and white mug in the microwave and close it",
        traj_sampling=True,
        final_step_reward=200.0  )

    iterator = buffer.iterator(batch_size=1)
    print("[INFO] Fetching one batch...\n")

    # -------------------------------
    # Your iterator returns:
    #   (Observation, actions)
    # -------------------------------
    output = next(iterator)
    # export_batch_to_json(output=output, task="robot_trajectory_prediction", texts="put both moka pots on the stove",)
    # exit()
    print("\n========== BATCH STRUCTURE ==========\n")
    print("output keys: ", output.keys())
    # print("prompt: ", output['prompt'])

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
    first_img = output['observations']['image'][0]    # [224,224,3]

    save_image_tensor_as_png(first_img, "first_base_rgb.png")
    # exit()
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

    # num_batches = 20
    # start = time.time()

    # for _ in range(num_batches):
    #     batch = next(iterator)

    # end = time.time()

    # avg = (end - start) / num_batches

    # print(f"\n⏱ Average time per batch over {num_batches} batches: {avg:.4f} seconds")
    # print(f"Total time: {end - start:.4f} seconds\n")
    import numpy as np
    import matplotlib.pyplot as plt

    def to_numpy(x):
        if x is None:
            return None
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)


    def select_trajectory(x, idx=0):
        if x is None:
            return None
        x = to_numpy(x)
        if x.ndim == 2:   # (B, T)
            return x[idx]
        elif x.ndim == 1: # (T,)
            return x
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")

    rewards = select_trajectory(output.get("rewards", None))
    mc_returns = select_trajectory(output.get("mc_returns", None))
    masks = select_trajectory(output.get("masks", None))


    print("\n------- REWARD SIGNALS -------")
    print("\n--MC Return of LAST timestep-- ", None if mc_returns is None else mc_returns[-1])
    print("\n--Mask of LAST timestep-- ", None if masks is None else masks[-1])
    print("\n--Reward of LAST timestep-- ", None if rewards is None else rewards[-1])

    if rewards is None:
        print("No rewards found in output.")
    else:
        print("rewards shape:", rewards.shape)
        print("mc_returns shape:", None if mc_returns is None else mc_returns.shape)
        print("masks shape:", None if masks is None else masks.shape)

        T = rewards.shape[0]
        t = np.arange(T)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # ---- Rewards ----
        axes[0].plot(t, rewards, label="reward", linewidth=2)
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Rewards vs Timestep")
        axes[0].grid(True)

        # ---- MC Returns ----
        if mc_returns is not None:
            axes[1].plot(t, mc_returns, label="mc_return", color="orange", linewidth=2)
            axes[1].set_ylabel("MC Return")
            axes[1].set_title("Monte-Carlo Returns vs Timestep")
            axes[1].grid(True)
        else:
            axes[1].set_visible(False)

        # ---- Masks ----
        if masks is not None:
            axes[2].step(t, masks, where="post", label="mask", linewidth=2)
            axes[2].set_ylabel("Mask")
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].set_title("Masks vs Timestep")
            axes[2].grid(True)
        else:
            axes[2].set_visible(False)

        axes[-1].set_xlabel("Timestep")

        import os
        import time

        # Directory to save plots
        plot_dir = "debug_plots"
        os.makedirs(plot_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plot_dir, f"-1_200_trajectory_rewards_{timestamp}.png")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"📈 Saved reward/return/mask plot to: {plot_path}")


