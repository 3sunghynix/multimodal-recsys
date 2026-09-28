import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

SETUP = '''# @title 0. 공통 준비 (모든 노트북 동일)
from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/yuhwani/multimodal-recsys.git"  # upstream에 머지되면 3sunghynix 주소로 변경
BRANCH = "main"
!rm -rf /content/repo && git clone -q -b {BRANCH} {REPO_URL} /content/repo
!pip -q install -r /content/repo/paper/requirements.txt
!cd /content/repo && git log -1 --format="코드 커밋: %h %s (%cd)"

import sys
sys.path.insert(0, "/content/repo/paper")
from msrs_paper.config import load_all, Paths
cfg, exps, plan = load_all()
DRIVE_ROOT = cfg["drive_root"]  # 공유 폴더 바로가기 경로가 다르면 이 줄만 바꾼다
paths = Paths(DRIVE_ROOT).make_dirs()
print("드라이브 폴더:", paths.root)
!nvidia-smi --query-gpu=name,memory.total --format=csv'''

VERIFY = '''# @title 1. 공유 준비물 확인 (체크섬이 다르면 멈춤)
from msrs_paper.utils import verify_manifest
m = verify_manifest(paths)
print(f"준비물 {len(m['files'])}개 일치 — 준비물 커밋 {m['git_commit']}")'''


def nb(title, intro, cells):
    all_cells = [{"cell_type": "markdown", "metadata": {}, "source": f"# {title}\n\n{intro}"}]
    for c in cells:
        if c.startswith("MD:"):
            all_cells.append({"cell_type": "markdown", "metadata": {}, "source": c[3:].strip()})
        else:
            all_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                              "source": c.strip()})
    for c in all_cells:
        lines = c["source"].split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cells": all_cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                             "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def save(name, notebook):
    (OUT / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


save("01_data_embeddings.ipynb", nb(
    "01. 데이터 준비 + 임베딩 (한 번만, 드라이브 소유자 계정으로)",
    "공유 폴더 소유자(결제한 계정)가 한 번만 실행한다. 원본은 Colab 로컬 디스크에만 받고, "
    "드라이브에는 CSV 2개와 축소 이미지 zip, 가공 파일만 올린다.\n\n"
    "준비: Colab 왼쪽 🔑(보안 비밀)에 `kaggle_username`, `kaggle_key` 등록 (새 방식 KGAT_ 토큰도 가능), H&M 대회 규칙 동의. "
    "런타임은 GPU.",
    [SETUP,
     '''# @title 1. Kaggle에서 원본 zip 받기 (Colab 로컬 디스크, 드라이브 아님)
import os, glob
from google.colab import userdata

def secret(*names):
    for n in names:
        try:
            v = userdata.get(n)
        except Exception:
            continue
        if v:
            return v.strip().strip('"').strip("'")
    raise KeyError(f"보안 비밀 {names} 없음")

user = secret("kaggle_username", "KAGGLE_USERNAME")
key = secret("kaggle_key", "KAGGLE_KEY")
for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
    os.environ.pop(k, None)
if key.startswith("KGAT_"):                 # 새 방식 API 토큰
    os.environ["KAGGLE_API_TOKEN"] = key
else:                                        # 기존 kaggle.json 방식
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key
print("사용자:", user, "| 키 형식:", "새 API 토큰" if key.startswith("KGAT_") else "기존 키", "| 길이:", len(key))

!pip -q install -U kaggle
!kaggle competitions files -c h-and-m-personalized-fashion-recommendations | head -5   # 로그인 확인
!df -h /content | tail -1
!mkdir -p /content/hm && kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p /content/hm
ZIP = glob.glob("/content/hm/*.zip")[0]
print(ZIP, round(os.path.getsize(ZIP) / 1e9, 1), "GB")''',
     '''# @title 2. CSV 2개만 드라이브 raw/ 로 풀기
!unzip -o -j "{ZIP}" articles.csv transactions_train.csv -d "{paths.raw}"
!ls -lh "{paths.raw}"''',
     '''# @title 3. 컬럼 값 확인 → analysis_plan.yaml 확정용
import pandas as pd
from msrs_paper import data_prep as dp
articles = pd.read_csv(paths.raw / "articles.csv", dtype={"article_id": "int64"})
for col in ["graphical_appearance_name", "product_group_name"]:
    print(articles[col].value_counts().to_string(), "\\n")
problems = dp.check_plan(articles, plan)
print("매핑에 없는 값:", problems or "없음")
print("analysis_plan status:", plan["status"])''',
     "MD: **여기서 멈추고** `paper/configs/analysis_plan.yaml`을 위 값으로 확정 → `status: final` → 커밋·푸시한 뒤, "
     "0번 셀부터 다시 실행한다 (1·2번은 파일이 있으면 건너뛰어도 됨). 결과를 보기 전에 확정해야 사전 등록이 된다.",
     '''# @title 4. 상품 표 + 분할 (main, alt)
import numpy as np
from msrs_paper import embeddings as em
from msrs_paper.utils import write_json
assert plan["status"] == "final", "analysis_plan.yaml을 확정(final)·커밋하고 0번 셀부터 다시 실행하세요"
assert not dp.check_plan(articles, plan)
image_ids = em.image_ids_in_zip(ZIP)
tx = dp.load_transactions(paths.raw / "transactions_train.csv")
print("구매 기록", f"{len(tx):,}", tx["t_dat"].min().date(), "~", tx["t_dat"].max().date())
items = dp.select_items(tx, dp.build_item_table(articles, plan), image_ids,
                        cfg["data"]["window_weeks"], cfg["data"]["alt_shift_weeks"])
items.to_parquet(paths.processed / "items.parquet", index=False)
items[["idx", "article_id"]].to_csv(paths.embeddings / "item_index.csv", index=False)
print("상품", len(items))
for split, shift in (("main", 0), ("alt", cfg["data"]["alt_shift_weeks"])):
    history, vt, tt, stats = dp.build_split(tx, items, cfg["data"]["window_weeks"], cfg["data"]["min_user_purchases"], shift)
    dp.save_split(paths.split_dir(split), history, vt, tt, stats)
    print(split, {k: v for k, v in stats.items() if k != "test_cells"})''',
     '''# @title 5. 필요한 상품 이미지만 축소해서 zip으로 (로컬 → 드라이브 복사)
IMG_ZIP = "/content/images_subset.zip"
em.write_image_subset(ZIP, items["article_id"].values, IMG_ZIP, cfg["data"]["image_max_side"])
!cp "{IMG_ZIP}" "{paths.raw}/images_subset.zip" && ls -lh "{paths.raw}"''',
     '''# @title 6. 임베딩 추출 → PCA → L2 정규화 (이미 있는 파일은 건너뜀, 런타임이 끊긴 뒤에도 이 셀부터 가능)
import os, shutil, torch
import numpy as np, pandas as pd
from msrs_paper import embeddings as em
IMG_ZIP = "/content/images_subset.zip"
if not os.path.exists(IMG_ZIP):
    shutil.copy(paths.raw / "images_subset.zip", IMG_ZIP)
items = pd.read_parquet(paths.processed / "items.parquet").sort_values("idx").reset_index(drop=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
E = cfg["embedding"]
if not (paths.embeddings / "text_minilm.npy").exists():
    raw = em.extract_text(items["text"].values, E["text_model"], 256, device)
    emb, info = em.reduce(raw, E["pca_dim"])
    em.save_embedding(paths, "text_minilm", emb, {**info, "model": E["text_model"]})
for name, spec in E["image_models"].items():
    if (paths.embeddings / f"img_{name}.npy").exists():
        continue
    raw = em.extract_image(IMG_ZIP, items["article_id"].values, spec, E["batch_size"], device)
    emb, info = em.reduce(raw, E["pca_dim"])
    em.save_embedding(paths, f"img_{name}", emb, {**info, **spec})
    print(name, emb.shape, info)''',
     '''# @title 7. 섞기 대응표 (전체 / 같은 품목 안) × seed
from msrs_paper.utils import write_json
for s in E["shuffle_seeds"]:
    g, t, info = em.make_shuffles(items["product_type_name"].values, s)
    np.save(paths.embeddings / f"shuffle_global_seed{s}.npy", g)
    np.save(paths.embeddings / f"shuffle_type_seed{s}.npy", t)
    write_json(paths.embeddings / f"shuffle_seed{s}.json", info)
    print(info)''',
     '''# @title 8. 임베딩 품질 눈으로 확인 (첫 열이 기준 상품, 나머지가 가까운 상품)
from IPython.display import Image, display
from msrs_paper.inspect_images import neighbor_grid
checks = paths.results / "checks"; checks.mkdir(parents=True, exist_ok=True)
for name in ["text_minilm"] + [f"img_{n}" for n in E["image_models"]]:
    out = neighbor_grid(items, np.load(paths.embeddings / f"{name}.npy"), IMG_ZIP, checks / f"neighbors_{name}.png")
    print(name); display(Image(str(out)))''',
     '''# @title 9. 체크섬 목록 작성 (이후 모든 노트북이 이걸로 같은 준비물인지 확인)
from msrs_paper.utils import write_manifest
m = write_manifest(paths)
print(len(m["files"]), "개 파일 등록 완료")'''
     ]))

save("02_tuning.ipynb", nb(
    "02. 하이퍼파라미터 튜닝 (한 번만)",
    "C2(텍스트 + Marqo) 조건, seed 42로 후보 설정을 검증 주에서 비교한다. 끝나면 가장 좋은 설정을 "
    "`paper/configs/paper.yaml`의 `train.default`에 반영하고 `tuned: true`로 바꿔 커밋·푸시한다.",
    [SETUP, VERIFY,
     '''# @title 2. 후보 설정 비교 (중단돼도 다시 실행하면 이어서)
import pandas as pd
from msrs_paper import two_tower
rows = []
for hp in cfg["train"]["candidates"]:
    info = two_tower.run(cfg, exps, paths, "C2", 42, "main", "tuning", mode="tune", hp_name=hp)
    rows.append({"hp": hp, **cfg["train"]["candidates"][hp], "valid_recall@300": info["valid_recall@300"],
                 "train_seconds": info["train_seconds"]})
table = pd.DataFrame(rows).sort_values("valid_recall@300", ascending=False)
table.to_csv(paths.results / "tuning.csv", index=False)
table''',
     "MD: 1회 학습 시간(`train_seconds`)도 확인한다. 실험 A는 13회라서, 1회가 30분을 넘으면 seed 44 묶음(C0·C2·C3·C4)을 "
     "다른 사람에게 옮기는 것을 검토한다."]))

save("03_train_group.ipynb", nb(
    "03. 실험 묶음 학습 (각자 자기 묶음)",
    "`GROUP`만 자기 묶음으로 바꿔 실행한다. A: 주 검정(Marqo, 13회) / B: ResNet-50(5회) / C: CLIP + 다른 테스트 주(9회).\n\n"
    "끝난 실험은 드라이브에 `run_info.json`이 생기고, 다시 실행하면 건너뛴다. Colab이 끊기면 0번 셀부터 다시 실행하면 이어서 돈다.",
    [SETUP, VERIFY,
     '''# @title 2. 내 묶음 선택
GROUP = "A"  # @param ["A", "B", "C"]
assert cfg["train"]["tuned"], "02_tuning 결과를 paper.yaml에 반영하고 tuned: true로 커밋한 뒤 실행하세요"
assert plan["status"] == "final"
plan_runs = [(b["split"], c, s) for b in exps["groups"][GROUP] for s in b["seeds"] for c in b["conds"]]
done = [r for r in plan_runs if (paths.run_dir(GROUP, r[0], f"{r[1]}_seed{r[2]}") / "run_info.json").exists()]
print(f"묶음 {GROUP}: 전체 {len(plan_runs)}회, 완료 {len(done)}회")''',
     '''# @title 3. 학습 실행 (로그는 logs/ 에도 저장)
from msrs_paper import two_tower
from msrs_paper.utils import now
log_file = paths.logs / f"group{GROUP}.log"
def log(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{now()} {msg}\\n")
two_tower.run_group(cfg, exps, paths, GROUP, log=log)
log(f"묶음 {GROUP} 완료")''']))

save("04_baselines.ipynb", nb(
    "04. 베이스라인 + 학습 없는 방식 (실험 C 담당)",
    "B1: 직전 1주 인기 / B2: 재구매 + 인기 / K0: 텍스트 이력 평균 최근접 / K2: 텍스트 + Marqo 이력 평균 최근접. "
    "결과는 `outputs/C/main/`에 Two-Tower와 같은 형식으로 저장된다.",
    [SETUP, VERIFY,
     '''# @title 2. 실행
from msrs_paper import baselines
for name in ["B1", "B2", "K0", "K2"]:
    baselines.run(cfg, paths, name, "main", group="C")''']))

save("05_analysis.ipynb", nb(
    "05. 채점 · 분석 · 표와 그림 (분석 담당)",
    "드라이브 `outputs/`에 모인 모든 결과를 같은 코드로 채점한다. 일부만 끝난 상태에서도 실행할 수 있고, "
    "새 결과가 올라오면 다시 실행하면 된다 (채점 결과는 `results/cache/`에 저장돼 재사용).",
    [SETUP, VERIFY,
     '''# @title 2. 전체 채점과 분석
from msrs_paper import analysis
t1, contrib, overall, mt = analysis.run_all(cfg, exps, plan, paths)
print("주 검정:", mt)
t1''',
     '''# @title 3. 기여도 표 (주 검정 조건)
m = plan["main_test"]
contrib[(contrib["measure"] == m["measure"]) & (contrib["segment"] == m["segment"])]''',
     '''# @title 4. 전체 이미지 기여도 (인코더 × 구간, 부트스트랩 95% 구간)
overall''',
     '''# @title 5. 그림 1 · 그림 2
from IPython.display import Image, display
display(Image(str(paths.results / "fig1.png")))
display(Image(str(paths.results / "fig2.png")))''',
     "MD: ## 라벨 표본 검사 (한 번만)\n외관 그룹별 이미지 40장 격자와 확인표(`results/label_check/label_check.csv`)를 만든다. "
     "격자를 보고 csv의 `label_correct`에 1/0을 채운 뒤 일치율을 논문에 적는다.",
     '''# @title 6. 라벨 표본 검사 자료 만들기
import shutil
from msrs_paper.data_prep import load_items
from msrs_paper.inspect_images import label_sample
IMG_ZIP = "/content/images_subset.zip"
shutil.copy(paths.raw / "images_subset.zip", IMG_ZIP)
out = paths.results / "label_check"; out.mkdir(parents=True, exist_ok=True)
label_sample(load_items(paths), IMG_ZIP, out)
!ls "{out}"''']))
save("06_run_all.ipynb", nb(
    "06. 전체 실행 (한 사람이 한 GPU로 A·B·C·D + 베이스라인 + 분석)",
    "모든 묶음을 한 세션에서 차례로 돌린다. 이미 끝난 실험은 건너뛰므로, 설정에 묶음이나 seed가 추가되면 "
    "이 노트북을 다시 모두 실행하면 새로 추가된 것만 돈다. **다른 사람이 같은 묶음을 동시에 돌리지 않게** 먼저 공지한다.\n\n"
    "D 묶음(인기도 입력)과 B·C의 seed 43·44는 결과 확인 후 추가한 분석이다 (사전 등록 아님).",
    [SETUP, VERIFY,
     '''# @title 2. 전체 진행 현황
assert cfg["train"]["tuned"] and plan["status"] == "final"
GROUPS = list(exps["groups"])
def status():
    for g in GROUPS:
        runs = [(b["split"], c, s) for b in exps["groups"][g] for s in b["seeds"] for c in b["conds"]]
        done = [r for r in runs if (paths.run_dir(g, r[0], f"{r[1]}_seed{r[2]}") / "run_info.json").exists()]
        print(f"묶음 {g}: 전체 {len(runs)}회, 완료 {len(done)}회")
status()''',
     '''# @title 3. 전체 학습 (끊기면 0번 셀부터 다시 모두 실행하면 이어서)
import time
from msrs_paper import two_tower
from msrs_paper.utils import now
log_file = paths.logs / "run_all.log"
def log(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{now()} {msg}\\n")
t0 = time.time()
for g in GROUPS:
    log(f"===== 묶음 {g} =====")
    two_tower.run_group(cfg, exps, paths, g, log=log)
    log(f"===== 묶음 {g} 완료 ({(time.time() - t0) / 60:.1f}분 경과) =====")
status()''',
     '''# @title 4. 베이스라인 B1 · B2, 학습 없는 방식 K0 · K2
from msrs_paper import baselines
for name in ["B1", "B2", "K0", "K2"]:
    baselines.run(cfg, paths, name, "main", group="C", log=log)''',
     '''# @title 5. 전체 채점 · 분석 (100회마다 진행 표시)
from msrs_paper import analysis
t1, contrib, overall, mt = analysis.run_all(cfg, exps, plan, paths, log=log)
print("\\n주 검정:", mt)
t1''',
     '''# @title 6. 전체 이미지 기여도 (인코더 × 구간)
overall''',
     '''# @title 7. 주 검정 조건의 층화 기여도 (패턴 − 무지)
m = plan["main_test"]
contrib[(contrib["measure"] == m["measure"]) & (contrib["segment"] == m["segment"])]''',
     '''# @title 8. 그림 1 · 그림 2
from IPython.display import Image, display
display(Image(str(paths.results / "fig1.png")))
display(Image(str(paths.results / "fig2.png")))''']))
print("ok", sorted(p.name for p in OUT.glob("*.ipynb")))
