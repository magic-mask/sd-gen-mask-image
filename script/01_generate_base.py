# 1. Base 이미지 생성 스크립트 (01_generate_base.py)
# 이 스크립트를 먼저 실행하여 100장의 기본 해상도(896x1152) 이미지와 설정값(JSON)을 하드 드라이브에 저장합니다.

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
OUTPUT_DIR = "./asian_mask_dataset"
NUM_IMAGES = 10
BASE_WIDTH = 1024 #896
BASE_HEIGHT = 1024 #1152
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────
# 모델 로드 (Base 전용)
# ──────────────────────────────────────────
print(f"[{MODEL_ID}] Base 파이프라인 로드 중...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(DEVICE)

# DPM++ 2M SDE Karras
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

# VRAM 최적화 (최신 문법 적용)
pipe.vae.enable_slicing()
pipe.enable_attention_slicing()

# ──────────────────────────────────────────
# 출력 폴더 (Base 이미지와 메타데이터 분리)
# ──────────────────────────────────────────
base_images_dir = os.path.join(OUTPUT_DIR, "base_images")
meta_dir = os.path.join(OUTPUT_DIR, "metadata")
os.makedirs(base_images_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# ──────────────────────────────────────────
# 속성 풀
# ──────────────────────────────────────────
genders = ["young East Asian man", "young East Asian woman"]
mask_types = ["blue surgical mask", "white KF94 mask", "black cloth face mask", "white disposable mask"]
hair_styles_male = ["natural short black hair", "slightly messy hair", "casual korean hairstyle"]
hair_styles_female = ["natural long black hair", "slightly messy hair", "casual straight hair", "simple tied hair"]
backgrounds = ["bathroom mirror", "subway station", "convenience store", "apartment hallway", "elevator interior", "office hallway", "underground parking lot", "street at night", "indoor cafe"]
lighting_conditions = ["natural window light", "fluorescent indoor lighting", "dim indoor lighting", "soft daylight", "harsh convenience store lighting", "phone camera flash", "overcast outdoor light"]
camera_styles = ["shot on smartphone", "Android phone camera photo", "casual selfie photo", "amateur everyday photography", "candid phone camera photo"]
angles = ["mirror selfie", "looking at camera", "slightly off-center framing", "candid composition"]
expressions = ["neutral expression", "relaxed eyes", "slightly tired expression", "natural expression"]

negative_prompt = (
    "anime, cartoon, illustration, painting, CGI, 3d render, "
    "beauty filter, flawless skin, airbrushed skin, plastic skin, "
    "studio lighting, professional photography, fashion magazine, "
    "oversharpened, HDR, overprocessed, perfect symmetric face, "
    "heavy bokeh, shallow depth of field, cinematic lighting, "
    "deformed face, distorted eyes, bad anatomy, lowres, blurry, "
    "watermark, text, logo"
)

def build_prompt():
    gender = random.choice(genders)
    mask = random.choice(mask_types)
    background = random.choice(backgrounds)
    lighting = random.choice(lighting_conditions)
    camera = random.choice(camera_styles)
    angle = random.choice(angles)
    expression = random.choice(expressions)
    
    hair = random.choice(hair_styles_female) if "woman" in gender else random.choice(hair_styles_male)

    prompt = (
        f"raw photo, candid photo of a {gender} wearing a {mask}, "
        f"{hair}, {expression}, {angle}, {background}, {lighting}, {camera}, "
        f"natural skin texture, subtle skin imperfections, slight film grain, "
        f"amateur smartphone photography, realistic everyday life"
    )

    return {
        "prompt": prompt, "gender": gender, "mask": mask, "background": background,
        "lighting": lighting, "camera": camera,
    }

# ──────────────────────────────────────────
# Base 생성 루프
# ──────────────────────────────────────────
print(f"총 {NUM_IMAGES}장 Base 이미지 생성 시작")

for i in tqdm(range(NUM_IMAGES)):
    data = build_prompt()
    #seed = random.randint(0, 99999999)
    seed = 12345 + i   # 시드 고정 (재현성 확보)
    generator = torch.Generator(DEVICE).manual_seed(seed)

    # Base generation
    image_base = pipe(
        prompt=data["prompt"],
        negative_prompt=negative_prompt,
        width=BASE_WIDTH,
        height=BASE_HEIGHT,
        guidance_scale=5.0,
        num_inference_steps=30,
        generator=generator,
    ).images[0]

    file_id = f"asian_mask_{i:05d}"
    image_path = os.path.join(base_images_dir, f"{file_id}.png")
    meta_path = os.path.join(meta_dir, f"{file_id}.json")

    image_base.save(image_path)

    metadata = {
        "image_id": file_id,
        "seed": seed,
        "prompt": data["prompt"],
        "negative_prompt": negative_prompt,
        "model": MODEL_ID,
        "scheduler": "DPM++ 2M SDE Karras",
        "base_resolution": [BASE_WIDTH, BASE_HEIGHT],
        "guidance_scale": 5.0,
        "base_steps": 30,
        "attributes": {
            "gender": data["gender"], "mask_type": data["mask"],
            "background": data["background"], "lighting": data["lighting"],
            "camera_style": data["camera"],
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

print("✅ Base 이미지 및 메타데이터 생성 완료")