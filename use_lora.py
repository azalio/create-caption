import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

# Use local LoRA model
LORA_PATH = "./my_first_flux_lora_v1.safetensors"

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32  # Use float32 for CPU compatibility
)

# Load LoRA weights directly from file
lora_weights = load_file(LORA_PATH)
pipe.unet.load_state_dict(lora_weights, strict=False)

# Generate image
prompt = "a futuristic FLUX meme in cyberpunk style"
image = pipe(prompt).images[0]

# Save result
image.save("flux_meme.png")
print("Image saved as flux_meme.png")
