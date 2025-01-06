import torch
from diffusers import StableDiffusionPipeline

# Use local LoRA model
LORA_PATH = "./my_first_flux_lora_v1.safetensors"

# Load base model and apply LoRA weights
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

pipe.unet.load_attn_procs(LORA_PATH)

# Generate image
prompt = "a futuristic FLUX meme in cyberpunk style"
image = pipe(prompt).images[0]

# Save result
image.save("flux_meme.png")
print("Image saved as flux_meme.png")
