"""Build the two-page, vector PDF handout for the FININ meeting.

Requires reportlab 4.4.9, pinned in requirements.txt:
    python scripts/07_build_meeting_handout.py

The paper-side statements were checked against Wang, Cohen and Ma (2024),
Sections 3-6, Figure 2, Tables 2-3 and Appendix A. Prototype statements use
the frozen reports/data_preparation.md and reports/results.md outputs.
"""

from __future__ import annotations

from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "FININ_pipeline_and_data_comparison.pdf"
PAGE_W, PAGE_H = landscape(A4)

NAVY = colors.HexColor("#12334A")
BLUE = colors.HexColor("#1D6C88")
ORANGE = colors.HexColor("#A96024")
INK = colors.HexColor("#1D303F")
MUTED = colors.HexColor("#5D7180")
RULE = colors.HexColor("#CFDCE3")
PALE_BLUE = colors.HexColor("#EFF6F8")
PALE_ORANGE = colors.HexColor("#FFF5EA")
PALE_ROW = colors.HexColor("#F7FAFB")


def font_setup() -> None:
    fonts = Path("C:/Windows/Fonts")
    if (fonts / "segoeui.ttf").exists():
        face_files = (fonts / "segoeui.ttf", fonts / "segoeuib.ttf", fonts / "segoeuii.ttf")
    else:
        bundled = Path(reportlab.__file__).resolve().parent / "fonts"
        face_files = (bundled / "Vera.ttf", bundled / "VeraBd.ttf", bundled / "VeraIt.ttf")
    for name, path in zip(("Segoe", "Segoe-Bold", "Segoe-Italic"), face_files):
        pdfmetrics.registerFont(TTFont(name, str(path)))


def lines_for(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        line = ""
        for word in paragraph.split():
            candidate = word if not line else f"{line} {word}"
            if pdfmetrics.stringWidth(candidate, font, size) <= width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                if pdfmetrics.stringWidth(word, font, size) > width:
                    raise ValueError(f"Unbreakable word exceeds column width: {word}")
                line = word
        if line:
            lines.append(line)
    return lines


def text_block(
    canvas: Canvas,
    text: str,
    x: float,
    top: float,
    width: float,
    *,
    font: str = "Segoe",
    size: float = 9.7,
    leading: float = 12.7,
    color=INK,
    max_height: float | None = None,
) -> float:
    lines = lines_for(text, font, size, width)
    used = len(lines) * leading
    if max_height is not None and used > max_height + 0.01:
        raise ValueError(f"Text overflows ({used:.1f} > {max_height:.1f}): {text}")
    canvas.setFillColor(color)
    canvas.setFont(font, size)
    for index, line in enumerate(lines):
        canvas.drawString(x, top - (index + 1) * leading, line)
    return used


def top_header(canvas: Canvas, kicker: str, title: str, subtitle: str, page: int) -> None:
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 10, PAGE_W, 10, fill=1, stroke=0)
    canvas.setFont("Segoe-Bold", 8.5)
    canvas.setFillColor(BLUE)
    canvas.drawString(39, PAGE_H - 37, kicker.upper())
    canvas.setFont("Segoe", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(PAGE_W - 40, PAGE_H - 37, "RESEARCH MEETING  |  04 OCT 2026")
    canvas.setFont("Segoe-Bold", 22)
    canvas.setFillColor(NAVY)
    canvas.drawString(39, PAGE_H - 66, title)
    canvas.setFont("Segoe", 10)
    canvas.setFillColor(MUTED)
    canvas.drawString(40, PAGE_H - 84, subtitle)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.7)
    canvas.line(39, PAGE_H - 95, PAGE_W - 39, PAGE_H - 95)
    canvas.setFont("Segoe", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(PAGE_W - 40, 19, f"{page} / 2")


def arrow(canvas: Canvas, x1: float, x2: float, y: float, color) -> None:
    canvas.setStrokeColor(color)
    canvas.setFillColor(color)
    canvas.setLineWidth(1.25)
    canvas.line(x1 + 1, y, x2 - 3, y)
    path = canvas.beginPath()
    path.moveTo(x2 - 3, y + 3)
    path.lineTo(x2, y)
    path.lineTo(x2 - 3, y - 3)
    path.close()
    canvas.drawPath(path, fill=1, stroke=0)


def lane_label(canvas: Canvas, bottom: float, height: float, label: str, detail: str, color) -> None:
    x, width = 39, 76
    canvas.setFillColor(color)
    canvas.roundRect(x, bottom, width, height, 8, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Segoe-Bold", 10)
    canvas.drawCentredString(x + width / 2, bottom + height / 2 + 17, label)
    text_block(canvas, detail, x + 9, bottom + height / 2 + 10, width - 18,
               size=8.1, leading=10.3, color=colors.white, max_height=35)


def stage_card(canvas: Canvas, x: float, bottom: float, width: float, height: float,
               step: str, title: str, body: str, accent, background) -> None:
    canvas.setFillColor(background)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.8)
    canvas.roundRect(x, bottom, width, height, 7, fill=1, stroke=1)
    canvas.setFillColor(accent)
    canvas.roundRect(x, bottom + height - 5, width, 5, 2, fill=1, stroke=0)
    canvas.setFont("Segoe-Bold", 8)
    canvas.drawString(x + 10, bottom + height - 22, step)
    text_block(canvas, title, x + 10, bottom + height - 25, width - 20,
               font="Segoe-Bold", size=9.8, leading=12.0, color=NAVY, max_height=25)
    canvas.setStrokeColor(RULE)
    canvas.line(x + 10, bottom + height - 52, x + width - 10, bottom + height - 52)
    text_block(canvas, body, x + 10, bottom + height - 58, width - 20,
               size=9.15, leading=11.65, max_height=height - 67)


def page_one(canvas: Canvas) -> None:
    top_header(
        canvas, "01 / computational pipeline", "FININ: paper and prototype",
        "A step-by-step map of the authors' model and the working student-scale implementation.", 1,
    )
    x0, card_w, gap, height = 124, 128, 8, 147
    paper_bottom, prototype_bottom = 313, 150
    paper = [
        ("01  INPUTS", "Market + news", "Index prices and descriptions; related companies. TRNA headlines with instrument-related sentiment."),
        ("02  ENCODE", "Fuse modalities", "Frozen language model and shared text layer; separate numeric encoders; market and news fusion MLPs."),
        ("03  RELATE", "News context", "Self-attention makes each news item aware of other news from the same day."),
        ("04  WEIGHT", "Market query", "Market representation queries news; attention gives each item a relative weight."),
        ("05  PREDICT", "Next-day trend", "Weighted news summary and recent market states enter an MLP; output is up or not-up."),
    ]
    prototype = [
        ("01  INPUTS", "Market + news", "SPY features and fixed description. Prior-day NIFTY headlines; generated financial sentiment."),
        ("02  ENCODE", "Fuse modalities", "Frozen BGE-small vectors and shared 384-to-64 layer; separate numeric and fusion MLPs."),
        ("03  RELATE", "News context", "One-head self-attention across at most 16 selected headlines."),
        ("04  WEIGHT", "Market query", "Market query and headline keys give masked attention weights."),
        ("05  PREDICT", "Next-day trend", "Weighted news summary and market state enter a small MLP; output is P(up)."),
    ]
    lane_label(canvas, paper_bottom, height, "PAPER", "Wang et al.\n(2024)", BLUE)
    lane_label(canvas, prototype_bottom, height, "OUR MODEL", "reduced\nFININ", ORANGE)
    for index, (step, title, body) in enumerate(paper):
        x = x0 + index * (card_w + gap)
        stage_card(canvas, x, paper_bottom, card_w, height, step, title, body, BLUE, PALE_BLUE)
        if index < 4:
            arrow(canvas, x + card_w, x + card_w + gap, paper_bottom + height / 2, BLUE)
    for index, (step, title, body) in enumerate(prototype):
        x = x0 + index * (card_w + gap)
        stage_card(canvas, x, prototype_bottom, card_w, height, step, title, body, ORANGE, PALE_ORANGE)
        if index < 4:
            arrow(canvas, x + card_w, x + card_w + gap, prototype_bottom + height / 2, ORANGE)

    canvas.setFillColor(colors.HexColor("#F5F8FA"))
    canvas.setStrokeColor(RULE)
    canvas.roundRect(39, 66, PAGE_W - 78, 69, 7, fill=1, stroke=1)
    canvas.setFillColor(NAVY)
    canvas.setFont("Segoe-Bold", 9.3)
    canvas.drawString(51, 115, "TIME ALIGNMENT")
    text_block(canvas, "Paper: market + news through day d (or a t-day lookback) -> whether close(d+1) > close(d).",
               51, 109, PAGE_W - 103, size=9.5, leading=12, max_height=14)
    text_block(canvas, "Prototype: headlines dated n -> SPY close on next session d -> whether close on following session q > close(d).",
               51, 89, PAGE_W - 103, size=9.5, leading=12, max_height=14)
    canvas.setFont("Segoe", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(40, 35, "Paper: [1] Figure 2 and Sections 3-5.  Prototype: [2] source modules listed in project architecture guide.")


TABLE_X = 39
TABLE_WIDTH = PAGE_W - 78
COL_WIDTHS = (145, 306, TABLE_WIDTH - 145 - 306)


def table_row(canvas: Canvas, bottom: float, height: float, label: str,
              paper_text: str, our_text: str, shade: bool) -> None:
    if shade:
        canvas.setFillColor(PALE_ROW)
        canvas.rect(TABLE_X, bottom, TABLE_WIDTH, height, fill=1, stroke=0)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.55)
    canvas.line(TABLE_X, bottom, TABLE_X + TABLE_WIDTH, bottom)
    x1 = TABLE_X + COL_WIDTHS[0]
    x2 = x1 + COL_WIDTHS[1]
    canvas.line(x1, bottom, x1, bottom + height)
    canvas.line(x2, bottom, x2, bottom + height)
    top = bottom + height - 4
    text_block(canvas, label, TABLE_X + 9, top, COL_WIDTHS[0] - 18,
               font="Segoe-Bold", size=9.1, leading=11.5, color=NAVY, max_height=height - 9)
    text_block(canvas, paper_text, x1 + 9, top, COL_WIDTHS[1] - 18,
               size=9.0, leading=11.4, max_height=height - 9)
    text_block(canvas, our_text, x2 + 9, top, COL_WIDTHS[2] - 18,
               size=9.0, leading=11.4, max_height=height - 9)


def page_two(canvas: Canvas) -> None:
    top_header(
        canvas, "02 / data and evaluation", "What changed - and why it matters",
        "The same next-session direction question is tested with different information and a smaller protocol.", 2,
    )
    top = 477
    header_h = 28
    canvas.setFillColor(NAVY)
    canvas.roundRect(TABLE_X, top - header_h, TABLE_WIDTH, header_h, 5, fill=1, stroke=0)
    canvas.setFont("Segoe-Bold", 9.3)
    canvas.setFillColor(colors.white)
    canvas.drawString(TABLE_X + 9, top - 19, "ASPECT")
    canvas.drawString(TABLE_X + COL_WIDTHS[0] + 9, top - 19, "PAPER [1]")
    canvas.drawString(TABLE_X + COL_WIDTHS[0] + COL_WIDTHS[1] + 9, top - 19, "PROTOTYPE [2]")
    rows = [
        (40, "Market and years", "S&P 500 and NASDAQ 100 indices; daily data from 2003-2018.",
         "SPY ETF proxy; NIFTY-covered news dates from 2010-2020."),
        (43, "News scale", "Reuters TRNA: over 2.7 million items; 24-1,953 items per day in Appendix A.",
         "NIFTY: 2,111 dated rows; 2,107 usable examples; 33,673 selected headline slots."),
        (40, "News sentiment", "TRNA probabilities for positive, neutral or negative influence on named instruments.",
         "TinyBERT-generated positive/neutral/negative probabilities for headline text."),
        (40, "Market input", "Six price/volume fields; market descriptions and related company names.",
         "Six normalized SPY channels; one fixed market description, no constituent list."),
        (40, "News timing", "Market and news through day d; lookbacks of 1, 3, 5, 10 and 20 days.",
         "No item timestamps: previous day's news + market through forecast close; one input day."),
        (38, "News selection", "Designed to use all available daily TRNA news items.",
         "At most 16 exact-deduplicated headlines per day, selected by a fixed hash."),
        (40, "Validation", "Ten sliding 500-day windows; chronological 8:1:1 split in each.",
         "One chronological split: 1,475 train / 315 valid / 317 test; 4 boundary exclusions."),
        (40, "Reported metrics", "Accuracy, profit-and-loss and Sharpe ratio; strongest claim is Sharpe improvement.",
         "Accuracy, balanced accuracy and log loss; no trading or Sharpe calculation."),
    ]
    cursor = top - header_h
    for index, (height, label, paper_text, our_text) in enumerate(rows):
        bottom = cursor - height
        table_row(canvas, bottom, height, label, paper_text, our_text, index % 2 == 1)
        cursor = bottom
    canvas.setStrokeColor(RULE)
    canvas.line(TABLE_X, top - header_h, TABLE_X + TABLE_WIDTH, top - header_h)
    canvas.setStrokeColor(RULE)
    canvas.rect(TABLE_X, cursor, TABLE_WIDTH, top - header_h - cursor, fill=0, stroke=1)

    canvas.setFillColor(PALE_ORANGE)
    canvas.setStrokeColor(colors.HexColor("#E6C49E"))
    canvas.roundRect(39, 78, TABLE_WIDTH, 43, 6, fill=1, stroke=1)
    canvas.setFont("Segoe-Bold", 9.4)
    canvas.setFillColor(ORANGE)
    canvas.drawString(51, 103, "INTERPRETATION")
    text_block(canvas,
               "Architecture reproduced; Sharpe claim untested. On 317 test dates: 58.4% accuracy vs 58.0% always-up, with 50.4% balanced accuracy.",
               51, 99, TABLE_WIDTH - 24, size=9.15, leading=11.3, max_height=25)
    canvas.setFont("Segoe", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(40, 54, "[1] Wang, Cohen and Ma, Findings of EMNLP 2024, Sections 3-6, Figure 2, Tables 2-3, Appendix A.")
    canvas.linkURL("https://aclanthology.org/2024.findings-emnlp.189/", (40, 50, 720, 66), relative=0)
    canvas.drawString(40, 41, "[2] This project's data_preparation.md, results.md and documented implementation (snapshot: 30 Sep 2026).")


def main() -> None:
    font_setup()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    canvas.setTitle("FININ - paper versus student-scale prototype")
    canvas.setAuthor("FININ research prototype")
    canvas.setSubject("Pipeline and dataset comparison for research meeting")
    page_one(canvas)
    canvas.showPage()
    page_two(canvas)
    canvas.showPage()
    canvas.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
