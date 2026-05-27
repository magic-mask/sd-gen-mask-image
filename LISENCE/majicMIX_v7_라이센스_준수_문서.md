# 라이센스 준수 확인서
## majicMIX realistic v7 상업적 활용
> CreativeML Open RAIL-M + Attachment B 기반 검토

---

| 항목 | 내용 |
|---|---|
| **문서 작성일** | 2026년 5월 27일 |
| **대상 모델** | digiplay/majicMIX_realistic_v7 |
| **모델 출처** | HuggingFace / CivitAI |
| **적용 라이센스** | CreativeML Open RAIL-M + Attachment B |
| **활용 목적** | AI 학습용 이미지 데이터셋 생성 및 상업적 활용 |

> ⚠ **주의사항**
> 본 문서는 majicMIX realistic v7 모델의 상업적 활용 시 준수해야 할 라이센스 조건을 분석하고, 법적 리스크를 최소화하기 위한 내부 가이드라인을 제공합니다.
> 법적 구속력이 있는 법률 자문을 대체하지 않으며, 중요한 결정 전 전문 법무 검토를 권장합니다.

---

## 1. 라이센스 개요

majicMIX realistic v7은 CreativeML Open RAIL-M 라이센스 하에 배포되며, Attachment B(추가 제한)가 별도로 부칙으로 적용됩니다. 이 라이센스는 오픈소스 활용과 책임 있는 사용의 균형을 목표로 설계되었습니다.

| 항목 | 내용 |
|---|---|
| 라이센스 유형 | CreativeML Open RAIL-M (2022.08.22 제정) |
| 저작권 주체 | Robin Rombach, Patrick Esser and contributors |
| HuggingFace 표기 | "other" (표준 목록 외 커스텀 조건 포함) |
| 베이스 모델 | Stable Diffusion 1.5 (Stability AI) |
| Attachment B 적용 | 모델 및 파생 모델 전체에 추가 제한 조항 적용 |

---

## 2. 상업적 활용 허용 범위

라이센스 원문 Section II는 저작권 및 특허 라이센스를 '영구적, 전 세계적, 비독점적, 무상, 무료, 취소 불가'로 부여합니다. 아래 표는 주요 사용 유형별 허용 여부를 정리합니다.

| 사용 유형 | 허용 여부 | 비고 |
|---|:---:|---|
| 모델로 이미지 생성 후 상업적 판매 | ✅ 허용 | Output은 사용자 소유 (Section III, 6항) |
| 생성 이미지를 AI 학습 데이터셋으로 활용 | ✅ 허용 | Output 사용에 별도 제한 없음 |
| 모델 파인튜닝 후 내부 상업 사용 | ✅ 허용 | 파생 모델 내부 활용은 자유 |
| 파인튜닝 모델을 무료로 공개 배포 | ⚠ 조건부 | 동일 라이센스(RAIL-M) + Attachment B 유지 필수 |
| 모델 자체를 유료로 판매·라이센싱 | ❌ 금지 | Attachment B 2항: Sale of the Model 명시 금지 |
| 모델을 병합(Merge)하여 유료 판매 | ❌ 금지 | Merge = Derivative 포함, 동일 금지 적용 |
| API 서비스로 모델 호스팅 후 유료 제공 | ⚠ 조건부 | 서비스(Output) 판매는 허용, 모델 라이센싱은 금지 |

> ✅ **핵심 원칙**
> 모델이 생성한 Output(이미지)은 사용자 소유이며 상업적 활용이 자유롭습니다.
> 단, 모델 가중치(Weights) 자체를 유상으로 판매하거나 라이센싱하는 행위는 명시적으로 금지됩니다.

---

## 3. 금지 사항 (Attachment A — Use Restrictions)

다음 목적으로의 모델 사용은 라이센스에 의해 명시적으로 금지됩니다. 생성된 Output도 동일하게 적용됩니다.

- 관련 법률을 위반하는 방식의 사용
- 미성년자를 착취하거나 해하는 콘텐츠 생성
- 타인을 해하기 위한 허위 정보 생성 및 배포
- 개인 식별 정보(PII) 생성 및 배포
- 명예훼손, 비방, 괴롭힘 콘텐츠 생성
- 법적 권리에 부정적 영향을 미치는 완전 자동화 의사결정
- 온·오프라인 행동이나 개인 특성 기반 차별적 활용
- 특정 집단의 취약점 착취를 통한 심리적·신체적 해악
- 법적으로 보호되는 특성에 기반한 차별
- 의료 조언 및 의료 결과 해석 제공
- 사법, 법 집행, 출입국 관련 의사결정 지원

---

## 4. Attachment B — 추가 제한 조항

Attachment B는 라이센스 본문에 추가되는 별도 조항으로, 모델 및 모든 파생 모델에 적용됩니다.

### 4-1. Changing Permissions (권한 변경 금지)

> ⚠ **조항 원문 요약**
> 파생 모델(Derivative)을 공유하거나 배포할 때, 원본 라이센스에 명시된 권한 조항을 수정하거나 삭제하는 것을 금지합니다.
> 즉, 파생 모델 배포 시 반드시 CreativeML Open RAIL-M + Attachment B를 그대로 유지해야 합니다.

### 4-2. Sale of the Model (모델 판매 금지)

> ❌ **판매/라이센싱 금지**
> 원문: *"Do not sell or license the Model in exchange for a fee or something else of value."*
> 모델(가중치, 체크포인트)을 금전 또는 그에 상응하는 대가를 받고 판매하거나 라이센싱하는 행위를 금지합니다.
> Merge(병합 모델)도 Derivative에 해당하므로 동일하게 적용됩니다.

---

## 5. 상업적 활용 시 준수 체크리스트

본 체크리스트는 법적 리스크를 최소화하기 위한 내부 점검용입니다. 각 항목을 확인 후 서명란에 날짜 및 담당자를 기록하세요.

| 확인 | 점검 항목 | 구분 | 확인자 / 날짜 |
|:---:|---|:---:|---|
| ☐ | 모델 가중치를 유료로 판매하거나 라이센싱하지 않는다 | 필수 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | 병합(Merge) 모델을 유료로 판매하지 않는다 | 필수 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | 파생 모델 배포 시 CreativeML Open RAIL-M + Attachment B를 유지한다 | 배포 시 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | 생성 이미지를 Attachment A 금지 목적에 사용하지 않는다 | 필수 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | 미성년자 관련 유해 콘텐츠를 생성하지 않는다 | 필수 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | 개인 식별 정보 생성 목적으로 사용하지 않는다 | 필수 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | 의료, 법 집행, 출입국 관련 의사결정에 활용하지 않는다 | 필수 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | API 서비스 제공 시 모델 자체가 아닌 Output(서비스)을 판매함을 명확히 한다 | 서비스 시 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | SD 1.5 기반 모델이므로 Stability AI 원본 모델 이용약관 별도 검토 | 권장 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |
| ☐ | 법적 분쟁 예방을 위해 전문 법무 자문 수행 여부 확인 | 권장 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / |

---

## 6. 결론 및 권고사항

majicMIX realistic v7을 활용한 이미지 생성 및 데이터셋 판매는 라이센스 범위 내에서 가능하며, 아래 조건을 준수하는 한 법적 리스크는 낮습니다.

> ✅ **최종 판단 요약**
>
> ① 생성된 이미지(Output) 판매: **허용** — 라이센스 Section III 6항에 의해 Output 소유권은 사용자에게 있음
>
> ② AI 학습 데이터셋 생성 및 판매: **허용** — Output 활용 제한 없음
>
> ③ 모델 가중치 자체 판매: **금지** — Attachment B 2항 명시
>
> ④ 파생/병합 모델 무료 배포: **허용** — 단, 동일 라이센스 유지 필수

### 추가 권고사항

- SD 1.5 원본 모델(Stability AI) 이용약관을 별도로 검토하여 추가 제한 여부를 확인하세요.
- 상업적 규모가 클 경우 IP 전문 변호사의 법률 자문을 병행하는 것을 권장합니다.
- CivitAI 원본 페이지(civitai.com/models/43331)의 Addendum 변경 사항을 주기적으로 모니터링하세요.
- 본 문서를 내부 규정 문서로 등록하고 관련 팀에 공유하여 조직 전체의 라이센스 인식을 높이세요.

---

## 7. 문서 확인 및 서명

| &nbsp; | 작성자 | 검토자 | 승인자 |
|---|---|---|---|
| **소속** | | | |
| **성명** | | | |
| **서명** | | | |
| **일자** | | | |

---

*majicMIX realistic v7 라이센스 준수 문서 | 작성일: 2026년 5월 27일*
