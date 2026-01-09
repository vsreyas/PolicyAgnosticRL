#!/usr/bin/env python3
"""Script to dump a tfrecord file as a gif by iterating over all images."""

import argparse
import os
import sys
from typing import List

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add the parent directory to path to import jaxrl_m modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxrl_m.data.image_replay_buffer_pi import ImageReplayBufferPi
from openpi.training.config import get_config


def create_gif_from_images(images: List[np.ndarray], output_path: str, duration: int = 100):
    """
    Create a GIF from a list of images.

    Args:
        images: List of numpy arrays representing images (H, W, 3) in uint8 format
        output_path: Path to save the GIF file
        duration: Duration of each frame in milliseconds
    """
    if not images:
        raise ValueError("No images provided to create GIF")

    # Convert numpy arrays to PIL Images
    pil_images = []
    for timestep, img in enumerate(images):
        # Ensure the image is in the correct format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Handle different image dimensions
        if img.ndim == 4:  # Batch dimension present
            img = img[0]

        # Convert to PIL Image
        pil_img = Image.fromarray(img)

        # Add timestep text overlay
        draw = ImageDraw.Draw(pil_img)
        text = f"Timestep: {timestep}"

        # Try to use a truetype font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()

        # Get text bounding box for background
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw semi-transparent background for text
        padding = 5
        background_box = [
            5,
            5,
            5 + text_width + 2 * padding,
            5 + text_height + 2 * padding
        ]
        draw.rectangle(background_box, fill=(0, 0, 0, 180))

        # Draw text in white
        draw.text((5 + padding, 5 + padding), text, fill=(255, 255, 255), font=font)

        pil_images.append(pil_img)

    # Save as GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )
    print(f"✅ GIF saved to {output_path}")


def dump_tfrecord_to_gif(
    tfrecord_path: str,
    output_gif_path: str,
    use_wrist_view: bool = True,
    use_language: bool = True,
    task_name: str = None,
    camera_view: str = "base",  # "base", "wrist", or "both"
    duration: int = 100,
    filter_successful_trajectories: bool = False,
):
    """
    Load a tfrecord file using ImageReplayBufferPi and dump it as a GIF.

    Args:
        tfrecord_path: Path to the tfrecord file
        output_gif_path: Path to save the output GIF
        use_wrist_view: Whether to use wrist view camera
        use_language: Whether to use language conditioning
        task_name: Optional task name to filter for
        camera_view: Which camera view to use ("base", "wrist", or "both")
        duration: Duration of each frame in milliseconds
        filter_successful_trajectories: Whether to filter for successful trajectories only
    """
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")

    # Prevent TensorFlow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Get PI config
    print("Loading PI-0.5 config...")
    pi_config = get_config("pi05_libero_custom_low_mem")
    pi_config.fsdp_devices = 1

    # Create ImageReplayBufferPi
    print(f"Loading tfrecord from {tfrecord_path}...")
    image_replay_buffer = ImageReplayBufferPi(
        data_paths=[tfrecord_path],
        seed=42,
        train=False,  # Set to False to avoid shuffling and repeating
        task_name=task_name,
        use_wrist_view=use_wrist_view,
        use_language=use_language,
        config=pi_config,
        final_step_sparse_reward=False,
        filter_successful_trajectories=filter_successful_trajectories,
        use_reverse_data_paths=False,
    )

    # Create iterator with batch_size=1 to get individual transitions
    print("Creating iterator...")
    iterator = image_replay_buffer.iterator(batch_size=1, training=False)

    # Collect all images
    base_images = []
    wrist_images = []

    print("Extracting images from tfrecord...")
    try:
        for batch_idx, batch in enumerate(tqdm(iterator)):
            # Extract images from observations
            if "observations" in batch and "image" in batch["observations"]:
                base_img = batch["observations"]["image"][0]  # Remove batch dimension
                base_images.append(base_img)

            if use_wrist_view and "observations" in batch and "wrist_image" in batch["observations"]:
                wrist_img = batch["observations"]["wrist_image"][0]  # Remove batch dimension
                wrist_images.append(wrist_img)

            # Break after collecting all unique timesteps
            # Since we're using train=False, the dataset won't repeat
    except (StopIteration, tf.errors.OutOfRangeError):
        # End of dataset
        pass

    print(f"Collected {len(base_images)} base camera images")
    if use_wrist_view:
        print(f"Collected {len(wrist_images)} wrist camera images")

    # Create GIFs based on camera_view selection
    if camera_view == "base" and base_images:
        create_gif_from_images(base_images, output_gif_path, duration)
    elif camera_view == "wrist" and wrist_images:
        create_gif_from_images(wrist_images, output_gif_path, duration)
    elif camera_view == "both" and base_images and wrist_images:
        # Create side-by-side view
        print("Creating side-by-side view...")
        combined_images = []
        for base_img, wrist_img in zip(base_images, wrist_images):
            # Ensure both images are uint8
            if base_img.dtype != np.uint8:
                if base_img.max() <= 1.0:
                    base_img = (base_img * 255.0).astype(np.uint8)
                else:
                    base_img = base_img.astype(np.uint8)

            if wrist_img.dtype != np.uint8:
                if wrist_img.max() <= 1.0:
                    wrist_img = (wrist_img * 255.0).astype(np.uint8)
                else:
                    wrist_img = wrist_img.astype(np.uint8)

            # Concatenate horizontally
            combined = np.concatenate([base_img, wrist_img], axis=1)
            combined_images.append(combined)

        create_gif_from_images(combined_images, output_gif_path, duration)
    else:
        print(f"❌ No images found for camera view: {camera_view}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump a tfrecord file as a GIF by iterating over all images"
    )
    parser.add_argument(
        "--tfrecord_path",
        type=str,
        help="Path to the input tfrecord file",
    )
    parser.add_argument(
        "--output_gif_path",
        type=str,
        help="Path to the output GIF file",
    )
    parser.add_argument(
        "--camera-view",
        type=str,
        default="base",
        choices=["base", "wrist", "both"],
        help="Which camera view to use (base, wrist, or both)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=100,
        help="Duration of each frame in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Optional task name to filter for",
    )
    parser.add_argument(
        "--no-wrist-view",
        action="store_true",
        help="Disable wrist view camera",
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Disable language conditioning",
    )
    parser.add_argument(
        "--filter-successful",
        action="store_true",
        help="Filter for successful trajectories only",
    )

    args = parser.parse_args()

    # Validate output path
    if not args.output_gif_path.endswith(".gif"):
        args.output_gif_path += ".gif"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_gif_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dump_tfrecord_to_gif(
        tfrecord_path=args.tfrecord_path,
        output_gif_path=args.output_gif_path,
        use_wrist_view=not args.no_wrist_view,
        use_language=not args.no_language,
        task_name=args.task_name,
        camera_view=args.camera_view,
        duration=args.duration,
        filter_successful_trajectories=args.filter_successful,
    )


if __name__ == "__main__":
    main()
