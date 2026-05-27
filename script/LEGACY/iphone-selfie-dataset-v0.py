import os
import json
import random

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
MODEL_ID = "SG161222/RealVisXL_V4.0"

OUTPUT_DIR = "./asian_mask_dataset"
NUM_IMAGES = 100

BASE_WIDTH = 896
BASE_HEIGHT = 1152

HR_WIDTH = 1344
HR_HEIGHT = 1728

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────
print(f"[{MODEL_ID}] 로드 중...")

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

# VRAM 최적화
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# Hires.fix용 img2img
pipe_hr = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe).to(DEVICE)

# ──────────────────────────────────────────
# 출력 폴더
# ──────────────────────────────────────────
images_dir = os.path.join(OUTPUT_DIR, "images")
meta_dir = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# ──────────────────────────────────────────
# 속성 풀
# ──────────────────────────────────────────
genders = [
    "young East Asian man",
    "young East Asian woman",
]

mask_types = [
    "blue surgical mask",
    "white KF94 mask",
    "black cloth face mask",
    "white disposable mask",
]

hair_styles_male = [
    "natural short black hair",
    "slightly messy hair",
    "casual korean hairstyle",
]

hair_styles_female = [
    "natural long black hair",
    "slightly messy hair",
    "casual straight hair",
    "simple tied hair",
]

backgrounds = [
    "bathroom mirror",
    "subway station",
    "convenience store",
    "apartment hallway",
    "elevator interior",
    "office hallway",
    "underground parking lot",
    "street at night",
    "indoor cafe",
]

lighting_conditions = [
    "natural window light",
    "fluorescent indoor lighting",
    "dim indoor lighting",
    "soft daylight",
    "harsh convenience store lighting",
    "phone camera flash",
    "overcast outdoor light",
]

camera_styles = [
    "shot on smartphone",
    "Android phone camera photo",
    "casual selfie photo",
    "amateur everyday photography",
    "candid phone camera photo",
]

angles = [
    "mirror selfie",
    "looking at camera",
    "slightly off-center framing",
    "candid composition",
]

expressions = [
    "neutral expression",
    "relaxed eyes",
    "slightly tired expression",
    "natural expression",
]

# ──────────────────────────────────────────
# 네거티브 프롬프트
# ──────────────────────────────────────────
negative_prompt = (
    "anime, cartoon, illustration, painting, CGI, 3d render, "
    "beauty filter, flawless skin, airbrushed skin, plastic skin, "
    "studio lighting, professional photography, fashion magazine, "
    "oversharpened, HDR, overprocessed, perfect symmetric face, "
    "heavy bokeh, shallow depth of field, cinematic lighting, "
    "deformed face, distorted eyes, bad anatomy, lowres, blurry, "
    "watermark, text, logo"
)

# ──────────────────────────────────────────
# 프롬프트 생성 함수
# ──────────────────────────────────────────
def build_prompt():

    gender = random.choice(genders)
    mask = random.choice(mask_types)

    background = random.choice(backgrounds)
    lighting = random.choice(lighting_conditions)

    camera = random.choice(camera_styles)

    angle = random.choice(angles)
    expression = random.choice(expressions)

    if "woman" in gender:
        hair = random.choice(hair_styles_female)
    else:
        hair = random.choice(hair_styles_male)

    prompt = (
        f"raw photo, candid photo of a {gender} wearing a {mask}, "
        f"{hair}, {expression}, "
        f"{angle}, "
        f"{background}, "
        f"{lighting}, "
        f"{camera}, "
        f"natural skin texture, "
        f"subtle skin imperfections, "
        f"slight film grain, "
        f"amateur smartphone photography, "
        f"realistic everyday life"
    )

    return {
        "prompt": prompt,
        "gender": gender,
        "mask": mask,
        "background": background,
        "lighting": lighting,
        "camera": camera,
    }

# ──────────────────────────────────────────
# 생성 루프
# ──────────────────────────────────────────
print(f"총 {NUM_IMAGES}장 생성 시작")

for i in tqdm(range(NUM_IMAGES)):

    data = build_prompt()

    seed = random.randint(0, 99999999)

    generator = torch.Generator(DEVICE).manual_seed(seed)

    # ─────────────────────
    # Step 1
    # Base generation
    # ─────────────────────
    image_base = pipe(
        prompt=data["prompt"],
        negative_prompt=negative_prompt,
        width=BASE_WIDTH,
        height=BASE_HEIGHT,
        guidance_scale=4.5,
        num_inference_steps=25,
        generator=generator,
    ).images[0]

    # ─────────────────────
    # Step 2
    # Hires.fix
    # ─────────────────────
    image_resized = image_base.resize(
        (HR_WIDTH, HR_HEIGHT),
        Image.LANCZOS
    )

    image_final = pipe_hr(
        prompt=data["prompt"],
        negative_prompt=negative_prompt,
        image=image_resized,
        strength=0.35,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=generator,
    ).images[0]

    # ─────────────────────
    # 저장
    # ─────────────────────
    file_id = f"asian_mask_{i:05d}"

    image_path = os.path.join(images_dir, f"{file_id}.png")
    meta_path = os.path.join(meta_dir, f"{file_id}.json")

    image_final.save(image_path)

    metadata = {
        "image_id": file_id,
        "seed": seed,
        "prompt": data["prompt"],
        "negative_prompt": negative_prompt,
        "model": MODEL_ID,
        "scheduler": "DPM++ 2M SDE Karras",
        "base_resolution": [
            BASE_WIDTH,
            BASE_HEIGHT
        ],
        "final_resolution": [
            HR_WIDTH,
            HR_HEIGHT
        ],
        "guidance_scale": 4.5,
        "base_steps": 25,
        "hires_steps": 20,
        "hires_strength": 0.35,
        "attributes": {
            "gender": data["gender"],
            "mask_type": data["mask"],
            "background": data["background"],
            "lighting": data["lighting"],
            "camera_style": data["camera"],
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

print("✅ 생성 완료")