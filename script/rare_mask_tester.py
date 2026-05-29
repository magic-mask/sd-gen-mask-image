# =========================================================
# Stable Mouth-Exposed Dataset Generator
# RealVisXL V5.0 optimized
# Focused latent-basin sampling
# =========================================================

import os
import json
import random
import torch

from tqdm import tqdm
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
MODEL_ID = "SG161222/RealVisXL_V5.0"

OUTPUT_DIR = "./mouth_exposed_dataset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WIDTH = 1024
HEIGHT = 1024

NUM_IMAGES = 200

GUIDANCE_SCALE = 5.8
STEPS = 30

# ──────────────────────────────────────────
# 프롬프트 고정
# ──────────────────────────────────────────
PROMPT = (
    "((mouth fully visible)), "
    "((mask pulled below mouth)), "
    "black cloth face mask, "
    "close-up raw smartphone selfie of a young East Asian woman, "
    "bathroom mirror, "
    "fluorescent indoor lighting"
)

NEGATIVE_PROMPT = (
    "properly worn mask, "
    "mask covering mouth, "
    "fully covered mouth, "
    "mask over lips, "
    "mask over nose, "
    "correct mask usage, "
    "worst quality, low quality, illustration, anime, cartoon, "
    "3d, painting, sketch, blurry, watermark, text, "
    "deformed face, bad anatomy"
)

# ──────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────
print(f"[{MODEL_ID}] loading pipeline...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to(DEVICE)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
)

pipe.enable_attention_slicing()
pipe.vae.enable_slicing()

# xformers 사용 가능 시 활성화
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers enabled")
except Exception:
    print("xformers not available")

# ──────────────────────────────────────────
# 출력 폴더
# ──────────────────────────────────────────
image_dir = os.path.join(OUTPUT_DIR, "images")
meta_dir = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(image_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# ──────────────────────────────────────────
# 생성 루프
# ──────────────────────────────────────────
print("\n===== generating mouth_exposed dataset =====")

for i in tqdm(range(NUM_IMAGES)):

    seed = random.randint(0, 999999999)

    generator = torch.Generator(DEVICE).manual_seed(seed)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=STEPS,
        generator=generator,
    ).images[0]

    file_id = f"mouth_exposed_{i:05d}"

    image_path = os.path.join(
        image_dir,
        f"{file_id}.png"
    )

    meta_path = os.path.join(
        meta_dir,
        f"{file_id}.json"
    )

    image.save(image_path)

    metadata = {
        "image_id": file_id,
        "class": "mouth_exposed",
        "seed": seed,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "model": MODEL_ID,
        "scheduler": "DPM++ 2M SDE Karras",
        "resolution": [WIDTH, HEIGHT],
        "guidance_scale": GUIDANCE_SCALE,
        "steps": STEPS,
        "attributes": {
            "gender": "young East Asian woman",
            "mask_type": "black cloth face mask",
            "background": "bathroom mirror",
            "lighting": "fluorescent indoor lighting",
            "camera_style": "close-up raw smartphone selfie",
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

print("\n✅ mouth_exposed dataset generation complete")