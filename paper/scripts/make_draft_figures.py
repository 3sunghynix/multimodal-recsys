"""논문 초안 그림 (docs/figures/). 수치 출처: docs/실험기록.md 7 · 8차 (seed 3개 평균 Recall@300).

그림 1: 이미지 사용 방식 세 가지의 구조도
그림 2: (v0.4에서 제외) 텍스트만 대비 개선율 막대그래프 — fig_improvement()로 다시 만들 수 있음

사용 (레포 루트에서): python paper/scripts/make_draft_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9})

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#ffffff", "#e6e5e1"
HILITE = "#2a78d6"          # 강조 (UME)
BASE = "#b9b8b2"            # 나머지 방식
BOX_TXT, BOX_IMG, BOX_MOD = "#e8f0fb", "#fdeedd", "#f0efec"

# ------------------------------------------------------------------ 그림 2 데이터
TEXT_ONLY = {"main": 0.0928, "s2": 0.0830, "s4": 0.0849}
METHODS = [  # (표시 이름, {주: R})
    ("이미지만", {"main": 0.1010, "s2": 0.0891, "s4": 0.0926}),
    ("함께 학습", {"main": 0.1056, "s2": 0.0935, "s4": 0.0967}),
    ("텍스트만 × 2 앙상블", {"main": 0.1070, "s2": 0.0955, "s4": 0.0973}),
    ("이미지만 × 2 앙상블", {"main": 0.1112, "s2": 0.0986, "s4": 0.1013}),
    ("함께 학습 × 2 앙상블", {"main": 0.1213, "s2": 0.1071}),
    ("따로 학습 후 결합 (UME)", {"main": 0.1292, "s2": 0.1151, "s4": 0.1177}),
]
WEEK_MARK = {"main": ("o", "main (09-16 ~ 22)"), "s2": ("s", "s2 (09-02 ~ 08)"), "s4": ("^", "s4 (08-19 ~ 25)")}


def fig_improvement():
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    names = [m[0] for m in METHODS]
    for i, (name, r) in enumerate(METHODS):
        pct = {w: (v / TEXT_ONLY[w] - 1) * 100 for w, v in r.items()}
        mean = sum(pct.values()) / len(pct)
        color = HILITE if "UME" in name else BASE
        ax.barh(i, mean, height=0.56, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
        for w, v in pct.items():
            ax.scatter(v, i, marker=WEEK_MARK[w][0], s=16, facecolor=SURFACE, edgecolor=INK2, linewidth=0.9, zorder=3)
        ax.text(max(pct.values()) + 1.0, i, f"+{mean:.0f}%", va="center", ha="left", fontsize=8.5,
                color=INK, fontweight="bold" if "UME" in name else "normal")
    ax.set_yticks(range(len(names)), names, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 46)
    ax.set_xlabel("텍스트만 대비 Recall@300 개선율 (%)", color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    handles = [plt.Line2D([], [], marker=m, linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=5, label=l)
               for m, l in WEEK_MARK.values()]
    ax.legend(handles=handles, title="테스트 주별 값 (막대는 평균)", loc="upper right", fontsize=7.5, title_fontsize=7.5,
              frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_improvement.png", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ 그림 1: 구조도
def box(ax, x, y, w, h, text, fc, ec=MUTED, bold=False, size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))


def fig_fusion():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    titles = ["(a) 함께 학습", "(b) 따로 학습 후 결합 (UME)", "(c) 앙상블 대조"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(t, fontsize=9, color=INK, pad=4, fontweight="bold" if "UME" in t else "normal")

    # (a) 텍스트 ⊕ 이미지 → 한 모델 → 점수
    a = axes[0]
    box(a, 0.05, 0.74, 0.40, 0.14, "텍스트 임베딩", BOX_TXT)
    box(a, 0.55, 0.74, 0.40, 0.14, "이미지 임베딩", BOX_IMG)
    box(a, 0.20, 0.47, 0.60, 0.14, "이어 붙이기 (512차원)", BOX_MOD)
    box(a, 0.20, 0.22, 0.60, 0.14, "Two-Tower 1개", BOX_MOD, bold=True)
    box(a, 0.30, 0.01, 0.40, 0.12, "추천 점수", SURFACE)
    arrow(a, 0.25, 0.74, 0.40, 0.61); arrow(a, 0.75, 0.74, 0.60, 0.61)
    arrow(a, 0.50, 0.47, 0.50, 0.36); arrow(a, 0.50, 0.22, 0.50, 0.13)

    # (b) 텍스트 → 모델 / 이미지 → 모델 → z 점수 합
    b = axes[1]
    box(b, 0.05, 0.74, 0.40, 0.14, "텍스트 임베딩", BOX_TXT)
    box(b, 0.55, 0.74, 0.40, 0.14, "이미지 임베딩", BOX_IMG)
    box(b, 0.05, 0.47, 0.40, 0.14, "텍스트 모델", BOX_TXT, bold=True)
    box(b, 0.55, 0.47, 0.40, 0.14, "이미지 모델", BOX_IMG, bold=True)
    box(b, 0.05, 0.22, 0.90, 0.14, "고객별 z 점수 합 (1:1)", BOX_MOD)
    box(b, 0.30, 0.01, 0.40, 0.12, "추천 점수", SURFACE)
    arrow(b, 0.25, 0.74, 0.25, 0.61); arrow(b, 0.75, 0.74, 0.75, 0.61)
    arrow(b, 0.25, 0.47, 0.40, 0.36); arrow(b, 0.75, 0.47, 0.60, 0.36); arrow(b, 0.50, 0.22, 0.50, 0.13)

    # (c) 같은 모델 × 2 (seed 다름) → z 점수 합
    c = axes[2]
    box(c, 0.05, 0.74, 0.90, 0.14, "같은 입력 (텍스트 · 이미지 · 둘 다)", BOX_MOD)
    box(c, 0.05, 0.47, 0.40, 0.14, "모델 (seed A)", BOX_MOD, bold=True)
    box(c, 0.55, 0.47, 0.40, 0.14, "모델 (seed B)", BOX_MOD, bold=True)
    box(c, 0.05, 0.22, 0.90, 0.14, "고객별 z 점수 합 (1:1)", BOX_MOD)
    box(c, 0.30, 0.01, 0.40, 0.12, "추천 점수", SURFACE)
    arrow(c, 0.30, 0.74, 0.25, 0.61); arrow(c, 0.70, 0.74, 0.75, 0.61)
    arrow(c, 0.25, 0.47, 0.40, 0.36); arrow(c, 0.75, 0.47, 0.60, 0.36); arrow(c, 0.50, 0.22, 0.50, 0.13)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig1_fusion.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_fusion()   # 그림 2(막대그래프)는 v0.4에서 제외: 표 1과 같은 수치, 3쪽 분량 맞추기
    print("저장:", sorted(p.name for p in OUT.glob("*.png")))
