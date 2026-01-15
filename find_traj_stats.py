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
import glob

def inspect_tfrecords(tfrecord_pattern):
    # TFRECORD_PATTERN = "/data/hf_cache/datasets/LIBERO/libero_10_tf/*.tfrecord"
    # TFRECORD_PATTERN = "/home/sreyas/vla/PolicyAgnosticRL/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/image_replay_buffer/episode_0.tfrecord"

    TFRECORD_PATTERN = tfrecord_pattern
    
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
        "vlm_output": tf.float32,
        "diffusion_actions": tf.float32,
        "next_state_vlm_output": tf.float32,
        "next_state_diffusion_actions": tf.float32,
        "valid_timesteps_for_action_chunk": tf.int32,
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
        # print("--- RAW FEATURE KEYS FOUND IN PROTO ---")
        # Parse the raw Example proto object to see exactly what keys exist on disk
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        found_keys = list(example.features.feature.keys())
        # for k in sorted(found_keys):
            # print(f" • {k}")
        # print("-" * 40 + "\n")

        # print("--- DECODING TENSORS ---")
        
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
                    # print(f"📝 [{key}] (String): {val}")
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
        return out['terminals'].numpy().sum()

if __name__ == "__main__":
    tfrecord_pattern = "/home/skowshik/vla/codebase/PolicyAgnosticRL/libero_10_pi05_put_the_two_mocha_pots_on_the_stove/results_expo/image_replay_buffer/*.tfrecord"
    # print(inspect_tfrecords(tfrecord_pattern))

    tfrecord_files = glob.glob(tfrecord_pattern)
    tfrecord_files = sorted(tfrecord_files)
    # breakpoint()

    successful_records = []
    failed_records = []
    for tfrecord_file in tfrecord_files:
        # breakpoint()
        # print(tfrecord_file)
        # print(inspect_tfrecords(tfrecord_file))
        if inspect_tfrecords(tfrecord_file) == 1:
            successful_records.append(os.path.basename(tfrecord_file))
        else:
            failed_records.append(os.path.basename(tfrecord_file))

    print(f"Successful records: {successful_records}")
    print(f"Failed records: {failed_records}")
