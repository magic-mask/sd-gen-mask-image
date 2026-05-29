# =========================================================
# Rare Mask State Dataset Generator (SDXL / RealVisXL)
# normal 제외 클래스 전용 생성 스크립트
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

OUTPUT_DIR = "./asian_mask_dataset"

BASE_WIDTH = 1024
BASE_HEIGHT = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 생성할 클래스 선택
TARGET_CLASSES = [
    "nose_exposed",
]

"""
"nose_exposed",
"under_chin",
"mouth_exposed",
"transition_putting_on",
"transition_taking_off",
"covering_one_side_only",
"fully_removed_holding",
"partial_removed",
"""



# 클래스별 생성 수
num_images = 10

# ──────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────
print(f"[{MODEL_ID}] 파이프라인 로드 중...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(DEVICE)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

pipe.vae.enable_slicing()
pipe.enable_attention_slicing()

# ──────────────────────────────────────────
# 속성 풀
# ──────────────────────────────────────────
genders = [
    "young East Asian man",
    "young East Asian woman"
]

mask_types = [
    "blue surgical mask",
    "white KF94 mask",
    "black cloth face mask",
    "white disposable mask"
]

hair_styles_male = [
    "natural short black hair",
    "slightly messy hair",
    "casual korean hairstyle"
]

hair_styles_female = [
    "natural long black hair",
    "slightly messy hair",
    "casual straight hair",
    "simple tied hair"
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
    "indoor cafe"
]

lighting_conditions = [
    "natural window light",
    "fluorescent indoor lighting",
    "dim indoor lighting",
    "soft daylight",
    "harsh convenience store lighting",
    "phone camera flash",
    "overcast outdoor light"
]

camera_styles = [
    "smartphone photo",
    "casual selfie",
    "amateur photo",
]

angles = [
    "mirror selfie",
    "looking at camera",
    "slightly off-center framing",
    "candid composition"
]

expressions = [
    "neutral expression",
    "relaxed eyes",
    "slightly tired expression",
    "natural expression"
]

# 어려운 상황용
difficulty_tags = [
    "wrinkled face mask",
    "creased mask fabric",
    "mask deformed from breathing",
    "uneven mask fit",
    "slightly collapsed mask shape",
    "mask folded near cheeks",
    "hand touching mask edge",
    "mask shadow on face",
    "partially occluded by hair",
    "wearing glasses",
    "earphone wires visible",
    "motion blur",
    "compression artifacts",
    "low light smartphone noise",
]

# ──────────────────────────────────────────
# 클래스별 핵심 프롬프트
# ──────────────────────────────────────────
CLASS_PROMPTS = {

    "nose_exposed":
        "surgical mask below nose, exposed nostrils, visible nose bridge",

    "under_chin":
        "face mask pulled under chin, exposed mouth and nose",

    "mouth_exposed":
        "mask below mouth, visible lips, exposed mouth",

    "transition_putting_on":
        "putting on face mask with hand, mask touching face, hand near cheek",

    "transition_taking_off":
        "removing face mask with hand, mask pulled away from face",

    "covering_one_side_only":
        "crooked face mask, one side uncovered, asymmetric fit",

    "fully_removed_holding":
        "holding face mask in hand near cheek, not wearing mask",

    "partial_removed":
        "mask hanging from one ear, partially removed",
}

# ──────────────────────────────────────────
# 클래스별 Negative Prompt
# ──────────────────────────────────────────
CLASS_NEGATIVE = {

    "nose_exposed":
        "fully covered nose, perfectly worn mask",

    "under_chin":
        "mask covering mouth and nose",

    "mouth_exposed":
        "fully covered mouth",

    "transition_putting_on":
        "perfectly worn mask, fully removed mask",

    "transition_taking_off":
        "perfectly worn mask",

    "covering_one_side_only":
        "symmetrical mask fit",

    "fully_removed_holding":
        "wearing mask on face",

    "partial_removed":
        "fully worn mask",
}

# 공통 negative
BASE_NEGATIVE = (
    "worst quality, low quality, illustration, 3d, 2d, painting, "
    "cartoons, sketch, plastic skin, wax skin, overprocessed, "
    "oversharpen, deformed eyes, asymmetrical eyes, "
    "malformed hands, extra fingers, bad anatomy, blurry, "
    "watermark, text"
)

# ──────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────
def build_prompt(class_name):

    gender = random.choice(genders)
    mask = random.choice(mask_types)
    background = random.choice(backgrounds)
    lighting = random.choice(lighting_conditions)
    camera = random.choice(camera_styles)
    angle = random.choice(angles)
    expression = random.choice(expressions)

    hair = (
        random.choice(hair_styles_female)
        if "woman" in gender
        else random.choice(hair_styles_male)
    )

    difficulty = ""

    if random.random() < 0.7:
        difficulty = ", " + random.choice(difficulty_tags)

    class_prompt = CLASS_PROMPTS[class_name]

    prompt = (
        f"ordinary smartphone selfie, casual phone camera photo, "
        f"raw photo, candid photo of a {gender}, "
        f"{class_prompt}, wearing a {mask}, "
        f"{hair}, {expression}, {angle}, "
        f"{background}, {lighting}, {camera}, "
        f"natural skin texture, subtle skin imperfections, "
        f"slight film grain, realistic everyday life"
        f"{difficulty}"
    )

    negative_prompt = BASE_NEGATIVE + ", " + CLASS_NEGATIVE[class_name]

    if class_name not in {"transition_taking_off", "transition_putting_on"}:
        negative_prompt += ", open mouth"

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "gender": gender,
        "mask": mask,
        "background": background,
        "lighting": lighting,
        "camera": camera,
    }

# ──────────────────────────────────────────
# 생성 루프
# ──────────────────────────────────────────
for class_name in TARGET_CLASSES:

    print(f"\n===== {class_name} 생성 시작 =====")

    class_dir = os.path.join(OUTPUT_DIR, class_name)
    image_dir = os.path.join(class_dir, "images")
    meta_dir = os.path.join(class_dir, "metadata")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    for i in tqdm(range(num_images)):

        data = build_prompt(class_name)

        seed = random.randint(0, 999999999)

        generator = torch.Generator(DEVICE).manual_seed(seed)

        guidance_scale = 5.0

        image = pipe(
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            width=BASE_WIDTH,
            height=BASE_HEIGHT,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

        file_id = f"{class_name}_{i:05d}"

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
            "class": class_name,
            "seed": seed,
            "prompt": data["prompt"],
            "negative_prompt": data["negative_prompt"],
            "model": MODEL_ID,
            "scheduler": "DPM++ 2M SDE Karras",
            "resolution": [BASE_WIDTH, BASE_HEIGHT],
            "guidance_scale": guidance_scale,
            "steps": 30,
            "attributes": {
                "gender": data["gender"],
                "mask_type": data["mask"],
                "background": data["background"],
                "lighting": data["lighting"],
                "camera_style": data["camera"],
            }
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=4
            )

print("\n✅ Rare mask state dataset generation complete")