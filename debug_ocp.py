import orbax.checkpoint as ocp
import pathlib

# Your path
path = pathlib.Path("/data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep29_bs64_v1_gradacc_2/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep29_bs64_v1_gradacc_2-v1/2000/params")

# Method 1: Use the PyTreeCheckpointHandler directly to read the structure
# This reads the keys without loading all the heavy arrays into memory
try:
    # This works for newer Orbax versions
    handler = ocp.PyTreeCheckpointHandler()
    # We pass None for the item because we just want to see what's on disk
    restored_structure = handler.restore(path, item=None)
    
    # Print the top-level keys
    print("Top level keys found:", restored_structure.keys())
    
    # Check for 'lora' or adapter specific keys deeper in the tree
    # If the dict is huge, just print a few keys to verify
    def print_keys(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print_keys(v, prefix + k + ".")
            else:
                if "lora" in prefix + k:
                    print(f"Found LoRA key: {prefix + k}")
                    return # Stop after finding one to confirm existence

    print("\nScanning for LoRA keys...")
    print_keys(restored_structure)

except Exception as e:
    print(f"Method 1 failed: {e}")
    # Method 2: Fallback for different Orbax versions (msgpack raw read)
    # Sometimes simplest is best to just see keys
    print("Attempting raw file listing...")
    import os
    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))
            break # Just show one file to confirm structure