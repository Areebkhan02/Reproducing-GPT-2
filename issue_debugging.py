# Load the checkpoint
import torch

# Import the necessary classes
from GPT2_Basic_Skeleton import GPT, GPTConfig, DataLoaderLite

checkpoint_path = "log/model_00100.pt"

checkpoint = torch.load(checkpoint_path)

# Load the configuration from the checkpoint
config = checkpoint['config']
print("config loaded")

# Initialize the model using the configuration
model = GPT(config)
print("model loaded")

print(checkpoint['model'].keys())

# # Remove the "_orig_mod." prefix from the keys
# state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}

# # Load the modified state dict
# missing_keys, unexpected_keys = model.load_state_dict(state_dict)
# #missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'])

# print("Missing keys:", missing_keys)
# print("Unexpected keys:", unexpected_keys)