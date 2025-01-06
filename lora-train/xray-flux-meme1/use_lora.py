import torch
from diffusers import FluxPipeline

# 1. Загружаем модель FLUX 
#    (bfloat16 предпочтительнее для FLUX, но при недостатке VRAM можно попробовать float16)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

# 2. Переносим на GPU (если у вас NVIDIA). 
#    Если на Apple Silicon, подставьте .to("mps"), либо используйте .to("cpu") (но будет медленно).
pipe.to("mps")

# 3. Включаем оффлоад модели на CPU — снижает пиковое использование VRAM
#    (Если у вас достаточно мощная видеокарта, можно убрать эту строку)
pipe.enable_model_cpu_offload()

# 4. Включаем attention slicing — тоже уменьшает пиковое использование VRAM при cross-attention
pipe.enable_attention_slicing("auto")

# 5. Загружаем LoRA-веса (пример: "my_first_flux_lora_v1.safetensors" из репо "azalio/meme1")
#    Если имя файла другое (my_first_flux_lora_v1_000001500.safetensors и т.д.) — укажите нужное.
pipe.load_lora_weights(
    repo_id_or_path="azalio/meme1",
    weight_name="my_first_flux_lora_v1.safetensors"
)

# 6. Задаём промпт и параметры генерации
prompt = "A funny meme cat wearing sunglasses, in a cartoon style"

# height/width — подбирайте под свою видеокарту; можно уменьшить, если не хватает VRAM.
image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=30,   
    max_sequence_length=512,   # рекомендуемый параметр для FLUX
    generator=torch.Generator("cuda").manual_seed(42)  # фиксируем seed для воспроизводимости
).images[0]

# 7. Сохраняем итоговое изображение
image.save("flux_lora_meme_cat.png")

# Дополнительно можно сгенерировать другое изображение с другим промптом:
another_prompt = "Meme cat with the text 'azalio-meme1'"
image2 = pipe(
    prompt=another_prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=30
).images[0]
image2.save("flux_lora_meme_cat2.png")


# 3. Генерация изображения
prompt = "meme cat"
image = pipeline(prompt, num_inference_steps=20, guidance_scale=4.0).images[0]
image.save("flux_lora_result.png")

prompt = "meme cat azalio-meme1"
image = pipeline(prompt, num_inference_steps=20, guidance_scale=4.0).images[0]
image.save("flux_lora_result2.png")