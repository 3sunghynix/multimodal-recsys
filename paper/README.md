# 논문 실험 (Colab + Google Drive)

"패션 추천에서 이미지는 어떤 상품에 도움이 되는가?" 실험 코드. 연구 설계는 [docs/논문계획서.md](../docs/논문계획서.md).
세 명이 각자 Colab GPU로 동시에 돌리고, 데이터와 결과는 공유 Google Drive 폴더 하나에 모인다.

## 1. 처음 한 번: 공유 폴더 연결

1. 폴더 소유자(김유환)가 드라이브의 `msrs_paper` 폴더를 김형우·김광석에게 **편집자**로 공유한다.
2. 김형우·김광석: 드라이브 → 공유 문서함 → `msrs_paper` 우클릭 → **바로가기 추가 → 내 드라이브**.
   Colab에서 `/content/drive/MyDrive/msrs_paper`로 보이면 성공. 경로가 다르면 노트북 0번 셀의 `DRIVE_ROOT`만 바꾼다.
3. 세 명 모두 Kaggle에서 [H&M 대회](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 규칙에 동의한다 (데이터를 직접 받지 않는 사람도).

드라이브 용량: 원본(약 30GB)은 드라이브에 올리지 않는다. 드라이브에는 CSV 2개(약 3.5GB), 축소 이미지 zip, 가공 파일,
결과만 쌓인다. 파일은 **만든 사람의 드라이브 용량**을 쓰므로 큰 준비물(01)은 폴더 소유자가 만든다. 실험 결과는 1회당 수십 MB다.

## 2. 실행 순서

| 순서 | 노트북 | 누가 | 하는 일 |
|---|---|---|---|
| 1 | `01_data_embeddings` | 폴더 소유자, 한 번 | Kaggle 원본 → 분할 → 축소 이미지 → 임베딩 4종 → 섞기 대응표 → `manifest.json` |
| 2 | `02_tuning` | 한 명, 한 번 | C2로 하이퍼파라미터 후보 비교 → `paper.yaml` 반영 + `tuned: true` 커밋 |
| 3 | `03_train_group` | **세 명 동시에** | 각자 `GROUP`(A / B / C)만 골라 실행 |
| 3 | `04_baselines` | 실험 C 담당 | 베이스라인 B1·B2, 학습 없는 방식 K0·K2 |
| 4 | `05_analysis` | 분석 담당 | 전체 채점 → 표 1, 기여도, 주 검정, 그림 1, 라벨 표본 검사 |

노트북은 GitHub에서 Colab으로 연다: `https://colab.research.google.com/github/yuhwani/multimodal-recsys/blob/main/paper/notebooks/<파일명>`
(upstream에 머지된 뒤에는 주소의 `yuhwani`를 `3sunghynix`로.) 런타임 유형은 GPU로 바꾼다.

### 01 중간에 멈추는 곳 (사전 등록)
3번 셀이 H&M 실제 컬럼 값을 보여준다. `configs/analysis_plan.yaml`의 매핑을 그 값으로 확정하고 `status: final`로 커밋·푸시한 뒤
0번 셀부터 다시 실행한다. **결과를 보기 전에** 확정해야 사전 등록이 된다.

## 3. 실험 묶음

| 묶음 | 담당 | 조건 | 횟수 | 논문에서 |
|---|---|---|---|---|
| A | 김유환 | C0·C2·C3·C4 × seed 42·43·44, C1 × 42 | 13 | 주 검정 (Marqo), 표 1, 그림 1 |
| B | 김형우 | C0·C5·C6·C7·C8 × 42 | 5 | 인코더 비교 (ResNet-50) |
| C | 김광석 | C0·C9~C12 × 42 + 다른 테스트 주 C0·C2·C3·C4 × 42 + 베이스라인 | 9 + 4 | 인코더 비교 (CLIP), 결과 신뢰도 |

조건 정의는 `configs/experiments.yaml`. 직접 빼서 비교하는 조건은 같은 묶음(같은 사람의 GPU)에 있다.
실험 A의 1회 학습이 30분을 넘으면 seed 44의 C0·C2·C3·C4를 **네 개 묶음째로** 다른 사람에게 넘긴다.

## 4. 결과를 합치기 위한 규칙

- 모든 실험은 팀 레포의 **같은 커밋**으로 돌린다. 코드가 바뀌면 0번 셀부터 다시 실행한다 (새로 clone).
- 1번 셀이 드라이브 준비물의 체크섬을 확인한다. 다르면 실험을 시작하지 않는다.
- 설정은 `configs/*.yaml`에서만 바꾸고 커밋한다. 노트북에서 숫자를 고치지 않는다.
- 결과 폴더 `outputs/{묶음}/{분할}/{조건}_seed{seed}/`에 `candidates.npy`, `users.npy`, `run_info.json`이 생긴다.
  `run_info.json`이 있으면 완료로 보고 다시 돌리지 않는다. Colab이 끊기면 그냥 다시 실행하면 이어서 돈다.
- 각자 채점하지 않는다. 분석 담당이 `05_analysis`로 한꺼번에 같은 코드로 채점한다.
- 남의 묶음 폴더에 쓰지 않는다. 결과를 지우거나 다시 돌려야 하면 팀에 먼저 알린다.

## 5. 드라이브 폴더 구조

```
msrs_paper/
├── raw/            articles.csv, transactions_train.csv, images_subset.zip
├── processed/      items.parquet, main/ alt/ (history, valid_targets, test_targets, stats.json)
├── embeddings/     text_minilm, img_marqo, img_resnet50, img_clipb32 (.npy + .json), shuffle_*.npy, item_index.csv
├── outputs/        A/ B/ C/ tuning/
├── results/        table1.csv, contributions.csv, main_test.json, fig1.png, cache/, checks/, label_check/
├── logs/           group{A,B,C}.log
└── manifest.json
```

## 6. 코드

| 파일 | 내용 |
|---|---|
| `msrs_paper/data_prep.py` | 상품 표, 시간 기준 분할, 정답 표 (신규·재구매는 전체 기록 기준) |
| `msrs_paper/embeddings.py` | 축소 이미지 zip, 텍스트·이미지 임베딩, PCA + L2 정규화, 섞기 대응표 |
| `msrs_paper/two_tower.py` | ID 없는 Two-Tower, 주 단위 학습 표본, 학습·추천·저장 |
| `msrs_paper/baselines.py` | B1·B2·K0·K2 |
| `msrs_paper/metrics.py` | Recall@K, NDCG@K, 고객별 평균 / 구매 건별 비율, 짝지은 부트스트랩 |
| `msrs_paper/analysis.py` | 전체 채점, 표 1, 층화 기여도(패턴 − 무지), 주 검정, 그림 1 |
| `msrs_paper/inspect_images.py` | 임베딩 최근접 이웃 격자, 라벨 표본 검사 |
| `notebooks/_make_notebooks.py` | 노트북 생성 스크립트. 노트북을 고칠 때는 이 파일을 고치고 `python _make_notebooks.py .`로 다시 만든다 |
