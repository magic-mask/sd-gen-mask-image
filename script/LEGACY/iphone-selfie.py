import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import os

# ──────────────────────────────────────────
# 모델 설정
# ──────────────────────────────────────────
MODEL_ID = "SG161222/RealVisXL_V4.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

# ✅ DPM++ 2M SDE Karras (노이즈 입자를 사진처럼 불규칙하게 분산)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

# ✅ 8GB VRAM 최적화
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# ✅ Hires.fix용 img2img — 가중치를 공유하므로 VRAM 추가 소모 없음
pipe_hr = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe).to("cuda")

# ──────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────
# SDXL은 Clip Skip이 없습니다 (아키텍처 자체가 다름)
# CFG Scale을 낮게 유지하는 것이 핵심 (7 이상이면 AI 느낌 살아남)

prompt = (
    "raw photo, candid mirror selfie of an ordinary East Asian young woman, "
    "holding smartphone, bathroom mirror, natural window light from the side, "
    "subtle film grain, slightly imperfect skin texture, "
    "effortless casual hair, muted tones, amateur everyday photography"
)

negative_prompt = (
    "masterpiece, best quality, highres, highly detailed, "
    "professional studio lighting, DSLR, 85mm lens, bokeh, "
    "flawless plastic skin, beauty filter, airbrushed, "
    "perfect symmetric face, oversharpened, CGI, artificial"
)

# ──────────────────────────────────────────
# 생성 루프
# ──────────────────────────────────────────
folder_name = "./test"
os.makedirs(folder_name, exist_ok=True)

for i in range(10):
    # ── Step 1: 베이스 이미지 생성 ──────────
    # SDXL 권장 해상도: 896×1152 (9:16 스마트폰 비율)
    image_base = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=4.5,      # ← 핵심: 낮을수록 사진 느낌
        width=896,
        height=1152,
    ).images[0]

    # ── Step 2: Hires.fix ──────────────────
    # diffusers의 SDXL img2img는 직접 업스케일을 하지 않음
    # → PIL로 먼저 리사이즈 후 img2img에 입력해야 함
    image_resized = image_base.resize(
        (1344, 1728),            # 1.5× upscale
        Image.LANCZOS
    )

    image_hr = pipe_hr(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_resized,
        num_inference_steps=20,
        guidance_scale=4.5,
        strength=0.35,           # 공식 권장: 0.2~0.5
    ).images[0]

    image_hr.save(f"{folder_name}/realvis_dataset_{i}.png")
    print(f"[{i+1}/10] 저장 완료")