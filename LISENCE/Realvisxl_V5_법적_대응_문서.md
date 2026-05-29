# RealVisXL V5.0 상업 사용 및 법적 대응 문서화

## 문서 목적

본 문서는 AI 셀카 애플리케이션 및 관련 머신러닝 데이터셋 개발 과정에서 사용되는 `SG161222/RealVisXL_V5.0` 모델의 라이선스, 사용 범위, 제한사항 및 향후 법적 분쟁 대응을 위해 작성되었다.

문서 작성 목적:

* 모델 사용 근거 기록
* 라이선스 준수 여부 기록
* 상업 사용 적법성 검토
* 데이터셋 생성 출처 기록
* 향후 법적 이슈 대응용 증빙 확보
* 내부 개발 문서화

---

# 1. 모델 정보

## 모델명

* RealVisXL V5.0

## 배포자

* SG_161222

## 배포 플랫폼

* CivitAI
* Hugging Face

## 모델 유형

* SDXL 기반 Checkpoint Merge
* Photorealistic 이미지 생성 모델

## Base Model

* SDXL Lightning

## 모델 목적

* Photorealistic image generation
* Human portrait generation
* Face mask synthetic dataset generation
* Segmentation dataset generation

---

# 2. 사용 목적

본 프로젝트에서 해당 모델은 다음 용도로 사용된다.

## 허용된 사용 범위

* Synthetic dataset generation
* Face mask segmentation dataset generation
* Inpainting training data generation
* AI selfie application development
* Commercial mobile application development
* SaaS/API inference service
* Research and development

## 실제 사용 방식

* Stable Diffusion XL inference
* 이미지 생성 전용 사용
* 모델 재배포 없음
* 모델 판매 없음
* merge 판매 없음
* 생성 결과물만 사용

---

# 3. 라이선스 정보

## 라이선스 명칭

CreativeML Open RAIL++-M

## 라이선스 출처

* CivitAI 모델 페이지
* Hugging Face 모델 카드
* LICENSE 파일

## 확인 일자

* 문서 작성 시점 기준 확인

---

# 4. 제작자 명시 허용 사항

CivitAI 모델 페이지 기준 제작자가 명시적으로 허용한 항목:

| 항목                                             | 상태 |
| ---------------------------------------------- | -- |
| Sell images they generate                      | 허용 |
| Run on services that generate for money        | 허용 |
| Run on CivitAI                                 | 허용 |
| Share merges using this model                  | 허용 |
| Have different permissions when sharing merges | 허용 |

제작자가 명시적으로 제한한 항목:

| 항목                                          | 상태 |
| ------------------------------------------- | -- |
| Sell this model or merges using this model  | 금지 |
| Use the model without crediting the creator | 제한 |

---

# 5. 본 프로젝트의 라이선스 준수 사항

본 프로젝트는 다음 사항을 준수한다.

## 준수 항목

* 모델 자체 판매 금지 준수
* safetensors 파일 재배포 금지
* merge 모델 판매 금지
* 생성 이미지 기반 서비스만 제공
* 모델 weight 공개 금지
* 원 제작자 정보 유지
* LICENSE 문서 보관

## 프로젝트 구조

본 프로젝트는 다음 구조를 따른다.

```text
RealVisXL V5.0
    ↓
Synthetic Image Generation
    ↓
Pseudo Segmentation GT 생성
    ↓
Segmentation Dataset 구축
    ↓
MobileNet 기반 경량 모델 학습
    ↓
온디바이스 추론
```

최종 배포 대상:

* 경량 segmentation model
* lightweight inpainting model
* mobile inference pipeline

본 프로젝트는 원본 Stable Diffusion weight를 최종 사용자에게 배포하지 않는다.

---

# 6. 데이터셋 생성 관련 문서화

## 생성 방식

Synthetic dataset은 아래 과정을 통해 생성된다.

1. RealVisXL V5.0 기반 이미지 생성
2. Rare mask state prompt generation
3. 자동 segmentation pseudo label 생성
4. 데이터 정제
5. 모델 학습

## 생성 클래스 예시

* nose_exposed
* under_chin
* mouth_exposed
* transition_putting_on
* transition_taking_off
* partial_removed

## 목적

* Face mask segmentation robustness 향상
* Rare state detection
* Mobile inference optimization

---

# 7. 재배포 정책

본 프로젝트는 아래 사항을 수행하지 않는다.

## 금지 대상

* RealVisXL weight 공개 배포
* safetensors 재판매
* checkpoint 판매
* merge 모델 판매
* weight extraction 제공

## 허용 범위

* 생성 이미지 사용
* synthetic dataset 사용
* 학습 결과물 사용
* 추론 서비스 제공
* 모바일 앱 서비스 제공

---

# 8. 생성 결과물 관련 해석

CreativeML Open RAIL++-M 라이선스 기준:

* Licensor claims no rights in the Output You generate.

따라서 생성 결과물(Output)은 프로젝트 측에서 활용 가능하다고 해석한다.

본 프로젝트는:

* 생성 이미지
* segmentation mask
* synthetic training pair

를 학습 데이터 용도로 사용한다.

---

# 9. 법적 리스크 관리

## 예상 가능한 리스크

### 1. 라이선스 변경

가능성:

* 배포자 정책 변경
* 플랫폼 정책 변경
* 모델 삭제
* 라이선스 수정

대응:

* LICENSE 원문 보관
* 모델 페이지 스크린샷 보관
* 모델 hash 기록
* 다운로드 시점 기록

---

### 2. 모델 재배포 이슈

대응:

* 원본 weight 미배포
* encrypted local storage 사용
* 추론 전용 구조 사용
* weight export 제한

---

### 3. 생성 이미지 저작권 문제

대응:

* synthetic dataset 생성 로그 저장
* seed 저장
* prompt 저장
* metadata 저장

---

### 4. NSFW 관련 정책 문제

대응:

* NSFW prompt 제한
* Safe prompt pipeline 사용
* App Store guideline 준수

---

# 10. 프로젝트 내부 기록 항목

다음 정보를 장기 보관 권장.

## 저장 권장 항목

* LICENSE 전문
* 모델 다운로드 날짜
* 모델 hash
* 모델 파일명
* 사용한 prompt
* 생성 seed
* metadata json
* 학습 config
* dataset generation code
* scheduler 설정
* inference parameter

---

# 11. 권장 보관 파일

```text
/legal
    /licenses
        realvisxl_license.txt

    /screenshots
        civitai_license.png
        huggingface_modelcard.png

    /hashes
        model_hash.txt

    /dataset_generation
        prompts.json
        seeds.json
        generation_config.json
```

---

# 12. 모델 사용 시점 기록

## 사용 모델

* SG161222/RealVisXL_V5.0

## 사용 목적

* Synthetic segmentation dataset generation
* Face restoration training
* Mobile AI selfie application research

## 사용 환경

* Diffusers
* PyTorch
* SDXL inference pipeline

## 사용 설정

* Scheduler: DPM++ SDE Karras
* CFG Scale: 5.0
* Steps: 30
* Resolution: 1024x1024

---

# 13. 최종 법적 해석 요약

현재 공개된 라이선스 및 제작자 정책 기준:

허용 가능성이 높은 영역:

* 생성 이미지 상업 활용
* SaaS/API 서비스
* 모바일 앱 서비스
* synthetic dataset 생성
* segmentation model 학습
* AI 셀카앱 개발

제한 가능성이 있는 영역:

* 모델 자체 판매
* checkpoint 재배포
* merge 판매
* weight 공유

본 프로젝트는 생성 결과물 및 학습 결과물만 사용하며, 원본 Stable Diffusion weight 자체를 재판매하거나 공개 배포하지 않는다.

---

# 14. 권장 추가 조치

## 강력 권장

* LICENSE 원문 백업
* 모델 페이지 PDF 저장
* 다운로드 시점 archive 저장
* civitai 페이지 스크린샷 저장
* 모델 hash 별도 보관

## 추가 권장

* 모델 버전 고정
* dependency lock
* inference reproducibility 기록
* 생성 로그 장기 저장

---

# 15. 참고

본 문서는 기술적/실무적 문서화 목적이며 법률 자문이 아니다.

상업 서비스 대규모 운영 시:

* 변호사 검토
* 기업 법무 검토
* 앱스토어 정책 검토

를 추가 권장한다.