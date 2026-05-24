import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import os
import random
import json
from tqdm import tqdm

def build_prompt_dataset(num_images=100, output_dir="./segmentation_dataset"):
    model_id = "RunDiffusion/Juggernaut-XL-v9"
    print(f"[{model_id}] 모델을 로드 중입니다. (해상도 1024x1024)...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

    # ✅ Fix 1: DPM++ 2M Karras로 교체 → realistic 품질 향상, step 효율 증가
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    ages = ["young adult", "middle-aged", "elderly", "teenager"]
    ethnicities = ["Asian", "Black", "Hispanic", "Caucasian", "Middle Eastern", "South Asian"]
    genders = ["male", "female", "person"]
    mask_types = ["blue surgical mask", "white N95 respirator", "black cloth mask",
                  "military camouflage mask", "patterned fabric mask", "green surgical mask"]
    lightings = [
        "natural window light",        # ✅ Fix 2: 인공 느낌 강한 수식어 제거
        "overcast outdoor light",
        "dim indoor ambient light",
        "soft directional light",
        "even indoor lighting",
        "shade outdoor light"
    ]
    # ✅ Fix 3: close-up headshot 제거 → hallucination 심화 방지
    camera_views = [
        "frontal portrait",
        "three-quarter profile",
        "slight side angle",
        "eye-level portrait",
        "slightly elevated angle",
        "candid street photo angle"
    ]
    backgrounds = [
        "plain white background",
        "indoor living room",
        "outdoor urban sidewalk",
        "hospital corridor",
        "office environment",
        "park setting"
    ]
    expressions = ["neutral expression", "relaxed expression", "looking forward", "eyes slightly averted"]

    # ✅ Fix 4: AI cliché 토큰 전부 제거 (8k / highly detailed / award winning / sharp focus)
    base_style = "documentary photograph, natural skin texture, DSLR photo, 50mm lens"

    # ✅ Fix 5: negative prompt에 AI artifact 억제 토큰 강화
    negative_prompt = (
        "cartoon, illustration, 3d render, CGI, painting, anime, "
        "blurred, lowres, bad anatomy, deformed, watermark, text, "
        "hat, sunglasses, oversaturated, HDR, overprocessed, plastic skin, "
        "artificial bokeh, heavy vignette, unnatural sharpening"
    )

    print(f"총 {num_images}장의 데이터셋 생성을 시작합니다...")

    for i in tqdm(range(num_images)):
        age       = random.choice(ages)
        ethnicity = random.choice(ethnicities)
        gender    = random.choice(genders)
        mask      = random.choice(mask_types)
        light     = random.choice(lightings)
        view      = random.choice(camera_views)
        bg        = random.choice(backgrounds)
        expr      = random.choice(expressions)

        prompt = (
            f"A {age} {ethnicity} {gender} wearing a {mask}, "
            f"{light}, {view}, {bg}, {expr}, {base_style}"
        )

        # ✅ Fix 6: CFG 4.5~7.0으로 낮춤 → 플라스틱 피부 / HDR 과증폭 방지
        cfg_scale  = round(random.uniform(4.5, 7.0), 1)
        # ✅ Fix 7: step 20~30으로 줄임 → DPM++ Karras에서 충분, overcooked 방지
        num_steps  = random.randint(20, 30)
        seed       = random.randint(1, 9_999_999)
        generator  = torch.Generator(device).manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=cfg_scale,
            height=1024,   # ✅ Fix 8: SDXL native 해상도 유지 (512 → 1024)
            width=1024,
            generator=generator
        ).images[0]

        file_prefix = f"mask_data_{i:05d}"
        image.save(os.path.join(images_dir, f"{file_prefix}.png"))

        metadata = {
            "image_id": file_prefix,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "inference_steps": num_steps,
            "scheduler": "DPMSolverMultistep_Karras",
            "attributes": {
                "age": age, "ethnicity": ethnicity, "gender": gender,
                "mask_type": mask, "lighting": light,
                "camera_view": view, "background": bg, "expression": expr
            }
        }
        with open(os.path.join(labels_dir, f"{file_prefix}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("✅ 데이터셋 구축 완료!")

if __name__ == "__main__":
    build_prompt_dataset(num_images=1000)