import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# 1. 모델 로드 (Juggernaut XL v9)
model_id = "RunDiffusion/Juggernaut-XL-v9"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16"
).to("cuda")

# 2. 견고한 학습을 위한 설정 (DPM++ 2M Karras 권장)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 3. 데이터 생성을 위한 프롬프트 전략
# 조명, 배경, 카메라 설정을 포함한 동적 프롬프트
prompt = """
(masterpiece, best quality:1.2), realistic photo of a person wearing a medical face mask, 
cinematic soft studio lighting, blurred office background, 
shot with 85mm lens, depth of field, sharp focus on eyes, 
high detailed skin texture, raw photo, color graded
"""

# 4. 이미지 생성 루프
for i in range(10): # 예시: 10장의 데이터 생성
    image = pipe(
        prompt=prompt,
        num_inference_steps=30, # Juggernaut XL v9 최적 스텝
        guidance_scale=5.0,     # 4~6 사이의 적절한 CFG
        width=832,              # 권장 종횡비
        height=1216
    ).images[0]
    
    image.save(f"mask_dataset_{i}.png")