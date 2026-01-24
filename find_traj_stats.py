from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import os
import numpy as np
import tensorflow as tf
import argparse

from absl import logging

import torch
import clip

import json
from openpi.training.config import get_config
import openpi.transforms as _transforms
import jax
from PIL import Image
import glob

def inspect_tfrecords(tfrecord_pattern):
    TFRECORD_PATTERN = tfrecord_pattern
    
    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/images1": tf.uint8,
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

    files = glob.glob(TFRECORD_PATTERN)
    files = sorted(files)[0]
    if not files:
        return None

    dataset = tf.data.TFRecordDataset(files)

    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        found_keys = list(example.features.feature.keys())

        features_dict = {
            k: tf.io.FixedLenFeature([], tf.string)
            for k in found_keys
        }
        
        parsed_features = tf.io.parse_single_example(raw_record, features_dict)

        out = {}
        
        for key in sorted(found_keys):
            raw_bytes = parsed_features[key]
            dtype = PROTO_TYPE_SPEC.get(key)
            
            try:
                if dtype is not None and dtype != tf.string:
                    tensor = tf.io.parse_tensor(raw_bytes, out_type=dtype)
                    out[key] = tensor
            except Exception:
                pass
        
        return out['terminals'].numpy().sum()

def main():
    parser = argparse.ArgumentParser(description="Inspect TFRecord files and classify trajectories as successful or failed.")
    parser.add_argument(
        '--tf_record_path',
        type=str,
        required=True,
        help="Glob pattern for TFRecord files (e.g., '/path/to/data/*.tfrecord')"
    )
    args = parser.parse_args()

    tfrecord_files = glob.glob(args.tf_record_path)
    tfrecord_files = sorted(tfrecord_files)

    if not tfrecord_files:
        print(f"No files found matching: {args.tf_record_path}")
        return

    successful_records = []
    failed_records = []
    
    for tfrecord_file in tfrecord_files:
        if inspect_tfrecords(tfrecord_file) == 1:
            successful_records.append(os.path.basename(tfrecord_file))
        else:
            failed_records.append(os.path.basename(tfrecord_file))

    print("=" * 50)
    print("TRAJECTORY SUMMARY")
    print("=" * 50)
    
    print(f"\nSuccessful trajectories: {len(successful_records)}")
    for record in successful_records:
        print(f"   {record}")
    
    print(f"\nFailed trajectories: {len(failed_records)}")
    for record in failed_records:
        print(f"   {record}")
    
    print("-" * 50)
    total = len(successful_records) + len(failed_records)
    print(f"Total: {total} trajectories")
    print(f"Success rate: {len(successful_records) / total * 100:.2f}%")

if __name__ == "__main__":
    main()