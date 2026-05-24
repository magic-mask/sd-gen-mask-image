# sd-gen-mask-image
Generate Images via Stable Diffusion

## Installation

```bash
# CUDA 설정
uv add torch torchvision --index https://download.pytorch.org/whl/cu132

# 의존성 패키지
uv add diffusers transformers accelerate
```

## 이미지 생성

```bash
uv run .\mask-image-gen\image-gen.py
```