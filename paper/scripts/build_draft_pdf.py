"""docs/논문초안.md → docs/논문초안_리뷰용.pdf (논문형 2단 배치, 리뷰 요청 절은 PDF에서 제외).

필요: pip install markdown, Chrome (또는 Edge) 헤드리스 인쇄.
사용 (레포 루트에서): python paper/scripts/build_draft_pdf.py
"""
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SRC = DOCS / "논문초안.md"
OUT = DOCS / "논문초안_리뷰용.pdf"
BROWSERS = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),          # Edge는 실행기가 분리돼 파일을 못 남기는 경우가 있어 Chrome 우선
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")]

CSS = """
@page { size: A4; margin: 13mm 13mm 13mm 13mm; }
body { font-family: 'Malgun Gothic', sans-serif; color: #0b0b0b; font-size: 9pt; line-height: 1.45; margin: 0; }
.banner { border-bottom: 1px solid #c9c8c2; padding: 0 0 4px; font-size: 7.4pt; color: #52514e; margin-bottom: 8px; text-align: center; }
.banner p { margin: 2px 0; }
h1 { font-size: 15pt; text-align: center; margin: 0 0 6px; line-height: 1.35; }
.authors { text-align: center; color: #52514e; margin-bottom: 6px; }
.abstract { margin: 0 5mm 8px; font-size: 8.9pt; text-align: justify; }
.abstract h2 { font-size: 10pt; text-align: center; margin: 0 0 4px; }
.cols { column-count: 2; column-gap: 6mm; text-align: justify; }
.cols h2 { font-size: 10.5pt; margin: 9px 0 4px; break-after: avoid; }
.cols h3 { font-size: 9.5pt; margin: 8px 0 3px; break-after: avoid; }
.cols p { margin: 0 0 5px; }
.cols ol, .cols ul { margin: 0 0 4px; padding-left: 14px; }
.wide { column-span: all; margin: 8px 0 10px; break-inside: avoid; }
.wide p.tcap, .narrow p.tcap { font-size: 8pt; margin: 0 0 3px; text-align: center; }
.narrow { margin: 6px 0 8px; break-inside: avoid; }
table { border-collapse: collapse; margin: 0 auto; font-size: 8.2pt; border-top: 1.4px solid #0b0b0b; border-bottom: 1.4px solid #0b0b0b; }
th, td { padding: 1.6px 9px; text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; }
.t3 table { font-size: 7.6pt; }
.t3 td:nth-child(4) { white-space: normal; }
th:first-child, td:first-child { text-align: left; padding-left: 4px; }
th { font-weight: 600; border-bottom: 0.8px solid #0b0b0b; }
tr.sep td { border-top: 0.6px solid #8a8984; }
tr.best td { background: #eef3fa; }
img { width: 74%; display: block; margin: 0 auto; }
.caption { font-size: 7.6pt; color: #333; margin-top: 2px; }
.refs { font-size: 7.3pt; }
.refs p { margin: 0 0 2px; }
code { font-size: 8pt; }
.appendix { break-before: page; }
.appendix h2 { font-size: 10.5pt; margin: 0 0 6px; }
.nw { white-space: nowrap; }
"""


LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s")


def fix_lists(text):
    """문단 바로 다음 줄에 오는 목록 앞에 빈 줄을 넣는다 (Python-Markdown은 빈 줄이 있어야 목록으로 인식)."""
    out = []
    for line in text.splitlines():
        if LIST_ITEM.match(line) and out and out[-1].strip() and not LIST_ITEM.match(out[-1]) and not out[-1].startswith(("|", "#")):
            out.append("")
        out.append(line)
    return "\n".join(out)


def md(text):
    return markdown.markdown(fix_lists(text), extensions=["tables"])


def wide_blocks(html):
    """표 · 그림(과 바로 뒤 캡션)을 2단 전체 폭으로. 열 3개 이하 표는 한 단 안에 둔다."""
    def table_block(m):
        cls = "narrow" if m.group(2).split("</tr>", 1)[0].count("<th") <= 3 else "wide"
        return f'<div class="{cls}"><p class="tcap">{m.group(1)}</p>{m.group(2)}</div>'
    html = re.sub(r"<p>(<strong>표 A?\d\.</strong>.*?)</p>\s*(<table>.*?</table>)", table_block, html, flags=re.S)
    html = html.replace('<div class="wide"><p class="tcap"><strong>표 A1.', '<div class="wide t3"><p class="tcap"><strong>표 A1.')   # 사전 등록 표: 긴 칸만 줄바꿈
    html = re.sub(r"<tr>(\s*<td>(?:텍스트만 × 2|이미지 드롭아웃 \(인코더))", r'<tr class="sep">\1', html)   # 표 안 묶음 구분선
    html = re.sub(r"<tr>(\s*<td><strong>)", r'<tr class="best">\1', html)                              # 강조 행
    html = re.sub(r'<p><img alt="[^"]*" src="([^"]+)" /></p>\s*<p>(<strong>그림 \d\.</strong>.*?)</p>',
                  r'<div class="wide"><img src="\1"><p class="caption">\2</p></div>', html, flags=re.S)
    return html


def build():
    s = SRC.read_text(encoding="utf-8")
    title_at = s.index("\n# ", s.index("\n") + 1)
    header = "\n".join(l[1:].strip() for l in s[:title_at].splitlines() if l.startswith(">"))
    paper, review = s[title_at:].split("## 리뷰 요청", 1)
    title, rest = paper.strip().split("\n", 1)
    authors, rest = rest.strip().split("\n", 1)
    abstract, body = rest.split("## 1. 서론", 1)
    body = "## 1. 서론" + body
    main, refs = body.split("## 참고문헌", 1)
    refs, appendix = (refs.split("## 부록 A", 1) + [""])[:2]   # 부록은 참고문헌 뒤 새 쪽
    clean = lambda t: t.replace("\n---\n", "\n")
    refs_html = "".join(f"<p>{markdown.markdown(l.strip())[3:-4]}</p>" for l in refs.strip().splitlines() if l.strip() and l.strip() != "---")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>{title.lstrip('# ').strip().replace(': ', ':<br>', 1)}</h1>
<div class="authors">{authors.strip()}</div>
<div class="abstract">{md(clean(abstract))}</div>
<div class="cols">{wide_blocks(md(clean(main)))}<h2>참고문헌</h2><div class="refs">{refs_html}</div></div>
{f'<div class="appendix"><h2>부록 A{appendix.split(chr(10), 1)[0]}</h2>{wide_blocks(md(clean(appendix.split(chr(10), 1)[1].strip().removesuffix('---'))))}</div>' if appendix else ""}
</body></html>"""
    html = html.replace('src="figures/', f'src="{(DOCS / "figures").as_uri()}/')
    browser = next(b for b in BROWSERS if b.exists())
    with tempfile.TemporaryDirectory() as tmp:   # 브라우저가 한글 출력 경로를 못 쓰는 경우가 있어 임시 경로에 먼저 저장
        page, pdf = Path(tmp) / "draft.html", Path(tmp) / "draft.pdf"
        page.write_text(html, encoding="utf-8")
        r = subprocess.run([str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                            f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf}", page.as_uri()],
                           capture_output=True, text=True, timeout=180)
        if not pdf.exists():
            raise RuntimeError(f"PDF 생성 실패: {r.returncode} {r.stderr[-500:]}")
        OUT.write_bytes(pdf.read_bytes())
    print("저장:", OUT, f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
