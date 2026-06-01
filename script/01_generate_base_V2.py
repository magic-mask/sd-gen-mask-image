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
BASE_WIDTH = 1024 
BASE_HEIGHT = 1024 
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

# VRAM 최적화
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
composition_buckets = {
    "viewpoint_and_angle": [
        "front view", "three-quarter view", "side profile view", 
        "high angle shot", "bird's eye view"
    ],
    "distance_and_framing": [
        "medium close-up", "mid-shot", "full-body shot", "distant view"
    ],
    "gaze_and_focus": [
        "looking at camera, sharp focus", 
        "looking to the side, slight motion blur background", 
        "candid photo style, natural look"
    ]
}

characters = [
    "raw photo of a korean person, 8k uhd", 
    "korean idol style portrait, highly detailed skin texture", 
    "photograph of a young east asian female", 
    "candid shot of a young adult male"
]
outfits_and_hair = [
    "wearing a casual hoodie", "school uniform", 
    "ponytail", "short haircut"
]

mask_buckets = {
    "colors": ["solid color white mask", "plain black surgical mask", "solid light pink medical mask"],
    "types": ["dental mask with visible pleats", "disposable medical mask", "kf94 mask"],
    "states": ["properly worn covering nose and mouth", "correctly fitting mask"]
}

backgrounds = [
    "simple clean classroom background, soft morning light", 
    "minimalist bedroom, subsurface scattering, indoor lighting", 
    "quiet botanical garden, natural daylight, depth of field", 
    "clean park background, golden hour lighting"
]

REALVIS_POSTFIX = "photorealistic, hyperrealistic, 35mm photograph, DSLR, soft lighting, detailed face"

# ──────────────────────────────────────────
# 로직 함수 (메타데이터 추적을 위해 내부 선택값도 함께 반환)
# ──────────────────────────────────────────
def generate_cohesive_composition(buckets, include_gaze=True):
    angle = random.choice(buckets["viewpoint_and_angle"])
    distance = random.choice(buckets["distance_and_framing"])
    
    prompt_elements = [angle, distance]
    gaze = None
    if include_gaze and random.random() > 0.3:
        gaze = random.choice(buckets["gaze_and_focus"])
        prompt_elements.append(gaze)
        
    return prompt_elements, {"angle": angle, "distance": distance, "gaze": gaze}


def build_prompt_pair():
    """
    포지티브/네거티브 프롬프트와 메타데이터용 추출 속성들을 함께 반환합니다.
    """
    pos_elements = []
    
    # 각 요소 선택
    char = random.choice(characters)
    outfit = random.choice(outfits_and_hair)
    
    mask_color = random.choice(mask_buckets["colors"])
    mask_type = random.choice(mask_buckets["types"])
    mask_state = random.choice(mask_buckets["states"])
    mask_str = f"{mask_color}, {mask_type}, {mask_state}"
    
    comp_elements, comp_meta = generate_cohesive_composition(composition_buckets)
    bg = random.choice(backgrounds)
    
    # 프롬프트 조립
    pos_elements.append(char)
    pos_elements.append(outfit)
    pos_elements.append(mask_str)
    pos_elements.extend(comp_elements)
    pos_elements.append(bg)
    pos_elements.append(REALVIS_POSTFIX)
    
    positive_prompt = ", ".join(pos_elements)
    
    # 네거티브 프롬프트
    neg_elements = [
        "deformed mask", "mask pulled down", "mask under chin", "double mask",
        "bad anatomy", "distorted facial features", "poorly drawn face", "extra limbs",
        "render", "3d", "cg", "cgi", "illustration", "anime", "cartoon", "painting", "drawing",
        "bad quality", "low quality", "blurry", "noise", "crowd", "multiple people",
        "text", "watermark", "signature"
    ]
    negative_prompt = ", ".join(neg_elements)
    
    # 하단 메타데이터 저장을 위한 속성 백업
    raw_attributes = {
        "character": char,
        "outfit_and_hair": outfit,
        "mask_style": mask_str,
        "composition": comp_meta,
        "background": bg
    }
    
    return {
        "positive": positive_prompt,
        "negative": negative_prompt,
        "attributes": raw_attributes
    }

# ──────────────────────────────────────────
# Base 생성 루프
# ──────────────────────────────────────────
print(f"총 {NUM_IMAGES}장 Base 이미지 생성 시작")

for i in tqdm(range(NUM_IMAGES)):
    data = build_prompt_pair()
    seed = 12345 + i   # 시드 고정
    generator = torch.Generator(DEVICE).manual_seed(seed)

    # Base generation (딕셔너리 키 및 변수명 에러 수정완료)
    image_base = pipe(
        prompt=data["positive"],
        negative_prompt=data["negative"],
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

    # 메타데이터 저장 에러 수정완료 (안전한 참조 구조로 변경)
    metadata = {
        "image_id": file_id,
        "seed": seed,
        "prompt": data["positive"],
        "negative_prompt": data["negative"],
        "model": MODEL_ID,
        "scheduler": "DPM++ 2M SDE Karras",
        "base_resolution": [BASE_WIDTH, BASE_HEIGHT],
        "guidance_scale": 5.0,
        "base_steps": 30,
        "attributes": data["attributes"]
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

print("✅ Base 이미지 및 메타데이터 생성 완료")