# 2. Hires.fix 전용 업스케일 스크립트 (02_hires_fix.py)
# 위 스크립트가 완전히 종료되어 VRAM이 초기화된 후 실행합니다. JSON 파일에서 동일한 프롬프트와 시드를 읽어와 img2img 파이프라인으로 처리합니다.

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
MODEL_ID = "SG161222/RealVisXL_V5.0"
OUTPUT_DIR = "./asian_mask_dataset"
HR_WIDTH = 1344
HR_HEIGHT = 1728
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────
# 모델 로드 (Img2Img 전용)
# ──────────────────────────────────────────
print(f"[{MODEL_ID}] Hires.fix (img2img) 파이프라인 로드 중...")
pipe_hr = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(DEVICE)

# DPM++ 2M SDE Karras
pipe_hr.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe_hr.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

# VRAM 최적화 (최신 문법 적용)
pipe_hr.vae.enable_slicing()
pipe_hr.enable_attention_slicing()

# ──────────────────────────────────────────
# 폴더 경로 설정
# ──────────────────────────────────────────
base_images_dir = os.path.join(OUTPUT_DIR, "base_images")
meta_dir = os.path.join(OUTPUT_DIR, "metadata")
final_images_dir = os.path.join(OUTPUT_DIR, "final_images")

os.makedirs(final_images_dir, exist_ok=True)

# ──────────────────────────────────────────
# Hires.fix 루프
# ──────────────────────────────────────────
# 메타데이터 파일 목록을 가져옵니다.
meta_files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
meta_files.sort()

print(f"총 {len(meta_files)}장 Hires.fix 업스케일링 시작")

for meta_file in tqdm(meta_files):
    meta_path = os.path.join(meta_dir, meta_file)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    file_id = metadata["image_id"]
    base_image_path = os.path.join(base_images_dir, f"{file_id}.png")
    final_image_path = os.path.join(final_images_dir, f"{file_id}.png")

    if not os.path.exists(base_image_path):
        print(f"⚠️ 경고: {base_image_path} 파일이 없습니다. 건너뜁니다.")
        continue

    # 1. Base 이미지 로드 및 리사이즈
    image_base = Image.open(base_image_path).convert("RGB")
    image_resized = image_base.resize((HR_WIDTH, HR_HEIGHT), Image.LANCZOS)

    # 2. 동일한 시드 생성
    generator = torch.Generator(DEVICE).manual_seed(metadata["seed"])

    # 3. Hires.fix (img2img)
    image_final = pipe_hr(
        prompt=metadata["prompt"],
        negative_prompt=metadata["negative_prompt"],
        image=image_resized,
        strength=0.35,
        guidance_scale=metadata["guidance_scale"],
        num_inference_steps=20,
        generator=generator,
    ).images[0]

    # 4. 최종 저장 및 메타데이터 업데이트
    image_final.save(final_image_path)

    metadata["final_resolution"] = [HR_WIDTH, HR_HEIGHT]
    metadata["hires_steps"] = 20
    metadata["hires_strength"] = 0.35

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

print("✅ Hires.fix 업스케일링 및 메타데이터 업데이트 완료")