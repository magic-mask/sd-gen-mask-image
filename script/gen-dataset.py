import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import os
import random
import json
from tqdm import tqdm

def build_prompt_dataset(num_images=100, output_dir="./segmentation_dataset"):
    model_id = "RunDiffusion/Juggernaut-XL-v9"
    print(f"[{model_id}] 모델을 로드 중입니다...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
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

    # ── 속성 풀 ────────────────────────────────────────────────
    ages        = ["young adult", "middle-aged", "elderly", "teenager"]
    ethnicities = ["Asian", "Black", "Hispanic", "Caucasian", "Middle Eastern", "South Asian"]
    genders     = ["male", "female", "person"]
    mask_types  = [
        "blue surgical mask", "white N95 respirator", "black cloth mask",
        "military camouflage mask", "patterned fabric mask", "green surgical mask"
    ]
    expressions = [
        "neutral expression", "relaxed expression",
        "looking forward",    "eyes slightly averted"
    ]
    backgrounds = [
        "plain white background", "indoor living room",
        "outdoor urban sidewalk", "hospital corridor",
        "office environment",     "park setting",
        "subway station",         "convenience store interior"
    ]

    # ── shot_type / lens / action 묶음 ─────────────────────────
    shot_configs = [
        ("close-up portrait",        "50mm lens", "looking at camera"),
        ("medium shot, waist up",    "35mm lens", "standing naturally"),
        ("medium shot, waist up",    "35mm lens", "talking to someone off-frame"),
        ("full body shot",           "24mm lens", "walking on sidewalk"),
        ("full body shot",           "24mm lens", "standing in environment"),
        ("environmental portrait",   "24mm lens", "sitting on bench"),
        ("over-the-shoulder shot",   "35mm lens", "looking away"),
        ("candid street photography","28mm lens", "walking through crowd"),
    ]

    # ── ✅ Fix 1: 카메라 디바이스 풀 분리 ──────────────────────
    # DSLR 고정 → 다양한 기기로 확장, 도메인 갭 감소
    camera_configs = [
        # (device_style,              dof_style,                        lighting_hint)
        ("shot on iPhone",            "everything in focus",             "selfie lighting"),
        ("selfie photo",              "flat focus, sharp background",    "front camera lighting"),
        ("candid photo",              "deep depth of field",             "natural light"),
        ("shot on Android phone",     "everything in focus",             "harsh overhead lighting"),
        ("CCTV footage style",        "flat focus, sharp background",    "low light, security camera"),
        ("DSLR photo, 35mm film",     "slight background blur",          "soft directional light"),
        ("mirrorless camera photo",   "moderate depth of field",         "even indoor lighting"),
        ("disposable camera photo",   "flat focus",                      "overcast outdoor light"),
    ]

    negative_prompt = (
        "cartoon, illustration, 3d render, CGI, painting, anime, "
        "blurred face, lowres, bad anatomy, deformed, watermark, text, "
        "hat, sunglasses, oversaturated, HDR, overprocessed, plastic skin, "
        # ✅ Fix 2: negative에 bokeh 명시 억제
        "heavy bokeh, artificial bokeh, heavy vignette, unnatural sharpening, "
        "shallow depth of field"
    )

    print(f"총 {num_images}장의 데이터셋 생성을 시작합니다...")

    for i in tqdm(range(num_images)):
        age        = random.choice(ages)
        ethnicity  = random.choice(ethnicities)
        gender     = random.choice(genders)
        mask       = random.choice(mask_types)
        bg         = random.choice(backgrounds)
        expr       = random.choice(expressions)

        shot_type, lens, action           = random.choice(shot_configs)
        device_style, dof_style, lighting = random.choice(camera_configs)

        # ✅ Fix 3: base_style에서 DSLR 제거 → camera_config에서 결정
        # ✅ Fix 4: dof_style을 프롬프트에 명시 → 아웃포커싱 수렴 방지
        prompt = (
            f"A {age} {ethnicity} {gender} wearing a {mask}, "
            f"{action}, {lighting}, {shot_type}, {bg}, {expr}, "
            f"{device_style}, {dof_style}, {lens}, "
            f"natural skin texture, documentary photograph"
        )

        cfg_scale = round(random.uniform(4.5, 7.0), 1)
        num_steps = random.randint(20, 30)
        seed      = random.randint(1, 9_999_999)
        generator = torch.Generator(device).manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=cfg_scale,
            height=1024,
            width=1024,
            generator=generator
        ).images[0]

        file_prefix = f"mask_data_{i:05d}"
        image.save(os.path.join(images_dir, f"{file_prefix}.png"))

        metadata = {
            "image_id":        file_prefix,
            "prompt":          prompt,
            "negative_prompt": negative_prompt,
            "seed":            seed,
            "cfg_scale":       cfg_scale,
            "inference_steps": num_steps,
            "scheduler":       "DPMSolverMultistep_Karras",
            "attributes": {
                "age":          age,
                "ethnicity":    ethnicity,
                "gender":       gender,
                "mask_type":    mask,
                "lighting":     lighting,
                "shot_type":    shot_type,
                "lens":         lens,
                "action":       action,
                "background":   bg,
                "expression":   expr,
                "device_style": device_style,
                "dof_style":    dof_style,
            }
        }
        with open(os.path.join(labels_dir, f"{file_prefix}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("✅ 데이터셋 구축 완료!")

if __name__ == "__main__":
    build_prompt_dataset(num_images=1000)