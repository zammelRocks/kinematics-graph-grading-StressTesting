"""Deterministic partial-credit sketches: turn a grading model's own structured
rubric-rule verdicts -- quantity, relation, expected threshold, and crucially
the *awarded/points partial-credit fraction* it actually gave for each rule
-- into ONE combined schematic plot per graded image.

This is intentionally NOT a reconstruction of the real submitted curve -- it
does not look at the image at all. It is a pure function of the rule dicts
already present in every grading record (see intermediate_step_rule_grades
.jsonl / .csv), so the same rules always produce the exact same plot: no
randomness, no model calls, nothing but arithmetic on numbers the grader
already reported.

The whole rubric (every rule, in rubric order) is sketched as ONE continuous
curve, built by concatenating one segment per rule -- each segment starting
exactly where the previous one ended, so the result reads as a single
timeline rather than disconnected fragments. Each segment is colored on a
red-to-green scale by that rule's actual `awarded / points` credit fraction:
fully unsatisfied segments are red, fully satisfied ones are green, and
partial credit falls in between -- so the unsatisfied part of the grade is
immediately visible as color, on one plot, instead of split across panels.

Used by deterministic_plotting.ipynb, which retrieves each sampled image's
"physics interpretation" (its list of graded rules) and passes the whole
list to plot_composite_grade() here.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image

POINTS_PER_SEGMENT = 100
# How far the schematic pushes a segment's level/slope/curvature away from
# `expected` at full credit -- proportional to the threshold's own scale,
# with a floor so an `expected == 0` rule (e.g. "parking", dfdx eq 0) still
# produces a visibly displaced, non-degenerate segment.
MARGIN_FRACTION = 0.3
MARGIN_FLOOR = 0.15

CREDIT_CMAP = colormaps["RdYlGn"]  # 0.0 -> red (unsatisfied), 1.0 -> green (full credit)


def credit_fraction(rule: Dict[str, Any]) -> float:
    """The actual credit the model gave, as a fraction of that rule's max
    points -- `awarded / points`, clipped to [0, 1]. This is the quantitative
    half of "partial credit from the model response vs. the actual credit":
    `points` is the rubric's maximum for this rule, `awarded` is what the
    model's response actually assigned."""
    points = float(rule.get("points") or 0.0)
    if points <= 0:
        return 0.0
    return float(np.clip(float(rule.get("awarded") or 0.0) / points, 0.0, 1.0))


def level_for_rule(relation: str, expected: float, fraction: float) -> float:
    """The deterministic target value (a level, slope, or curvature,
    depending on the rule's `quantity`) that reflects how close the model's
    *actual* awarded credit fraction is to fully satisfying
    `quantity {relation} expected`. `fraction=1.0` reproduces the fully-
    satisfying level, `fraction=0.0` the fully-violating level, and anything
    in between interpolates linearly -- so a 1/3-credit rule sketches a
    segment one third of the way from "wrong" to "right", not a binary flip.
    """
    margin = max(MARGIN_FRACTION * abs(expected), MARGIN_FLOOR)
    if relation in ("gt", "ge"):
        return expected + margin * (2 * fraction - 1)
    if relation in ("lt", "le"):
        return expected - margin * (2 * fraction - 1)
    if relation == "eq":
        return expected + margin * (1 - fraction)
    raise ValueError(f"unknown relation: {relation!r}")


def _segment_curve(quantity: str, level: float, y_start: float, slope_start: float, n: int = POINTS_PER_SEGMENT):
    """One rule's local segment, continuing smoothly from the previous
    segment's end value (and, for curvature rules, end slope). Returns
    (x_local in [0, 1], y_local, y_end, slope_end)."""
    x_local = np.linspace(0.0, 1.0, n)
    if quantity == "f":
        # "f" rules assert a value/level, not a slope -- ease from wherever
        # the timeline currently sits toward that level, then hover there.
        y_local = y_start + (level - y_start) * (1 - np.exp(-5 * x_local)) + 0.02 * np.sin(4 * np.pi * x_local)
        slope_end = 0.0
    elif quantity == "dfdx":
        y_local = y_start + level * x_local
        slope_end = level
    elif quantity == "d2fdx2":
        y_local = y_start + slope_start * x_local + 0.5 * level * x_local**2
        slope_end = slope_start + level
    else:
        raise ValueError(f"unknown quantity: {quantity!r}")
    return x_local, y_local, float(y_local[-1]), slope_end


def build_composite_curve(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One continuous, deterministic schematic curve built by concatenating
    one segment per rubric rule, in rubric order -- each segment's shape
    driven only by that rule's own quantity/relation/expected/awarded-credit
    fraction, and continuing smoothly from the previous segment's endpoint.
    Returns a list of {"rule", "fraction", "x", "y"} dicts, one per rule."""
    segments = []
    y_start, slope_start = 0.0, 0.2
    for i, rule in enumerate(rules):
        fraction = credit_fraction(rule)
        level = level_for_rule(rule["relation"], float(rule["expected"]), fraction)
        x_local, y_local, y_end, slope_end = _segment_curve(rule["quantity"], level, y_start, slope_start)
        segments.append({"rule": rule, "fraction": fraction, "x": i + x_local, "y": y_local})
        y_start, slope_start = y_end, slope_end
    return segments


def plot_composite_grade(ax: plt.Axes, record: Dict[str, Any]) -> None:
    """Draw ONE combined schematic for the whole graded image: every rubric
    rule as one segment of a single continuous timeline, colored red
    (unsatisfied) to green (full credit) by its actual awarded fraction, with
    each segment labeled by rule id and credit percentage."""
    rules: List[Dict[str, Any]] = record["rules"]
    segments = build_composite_curve(rules)

    for seg in segments:
        color = CREDIT_CMAP(seg["fraction"])
        ax.plot(seg["x"], seg["y"], color=color, linewidth=3.4, solid_capstyle="round", zorder=3)
        if seg is not segments[0]:
            ax.axvline(seg["x"][0], color="#bdbdbd", linewidth=0.8, linestyle=":", zorder=1)
        rule = seg["rule"]
        verdict = "✓" if rule["satisfied"] else "✗"
        mid_idx = len(seg["x"]) // 2
        ax.annotate(
            f"{rule['id']}\n{verdict} {seg['fraction']*100:.0f}%",
            xy=(seg["x"][mid_idx], seg["y"][mid_idx]), xytext=(0, 13), textcoords="offset points",
            ha="center", fontsize=7.5, fontweight="bold", color=color, zorder=4,
        )

    ax.set_xticks([])
    ax.set_ylabel("schematic trajectory (illustrative, not to scale)", fontsize=8.5)
    ax.set_title(
        f"Grade schematic: {record['total_awarded']:g}/{record['max_score']:g} pts "
        f"(score_ratio={record['score_ratio']:.2f})",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ymin, ymax = ax.get_ylim()
    pad = 0.15 * (ymax - ymin or 1.0)
    ax.set_ylim(ymin - pad, ymax + 2.2 * pad)  # room for the segment labels above the curve

    sm = plt.cm.ScalarMappable(cmap=CREDIT_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.055, pad=0.1)
    cbar.set_label("awarded credit fraction  (red = unsatisfied → green = full credit)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def _wrap_rule_captions(rules: List[Dict[str, Any]], width: int = 100) -> List[str]:
    lines: List[str] = []
    for rule in rules:
        verdict = "SATISFIED" if rule["satisfied"] else "NOT SATISFIED"
        header = (
            f"{rule['id']} ({rule['quantity']} {rule['relation']} {rule['expected']:g}) -- {verdict}, "
            f"awarded {rule.get('awarded', 0):g}/{rule.get('points', 0):g} pts"
        )
        lines.extend(fill(header, width=width).split("\n"))
        observed = rule.get("observed", "") or ""
        if observed:
            lines.extend(fill("Observed: " + observed, width=width - 2).split("\n"))
        reason = rule.get("partial_credit_reason", "") or ""
        if reason:
            lines.extend(fill("Credit reason: " + reason, width=width - 2).split("\n"))
        lines.append("")
    return lines


def build_sample_figure(record: Dict[str, Any], out_path: Path) -> Path:
    """One combined figure for a single graded image, three parts stacked
    vertically: (1) the trajectory's full panel.png strip for context, if
    present, (2) the real graded image next to the ONE composite grade
    schematic built purely from its rules' quantity/relation/expected/
    awarded/points fields, and (3) each rule's own observed/credit-reason
    text, for manual verification against both the schematic and the image.
    """
    rules: List[Dict[str, Any]] = record["rules"]
    img_path = Path(record["generated_image"])
    panel_path = img_path.parent / "panel.png"
    has_panel = panel_path.exists()

    caption_lines = _wrap_rule_captions(rules)
    panel_row_h = 2.1
    content_row_h = 5.0
    caption_row_h = max(1.6, 0.205 * len(caption_lines) + 0.4)

    row_heights = ([panel_row_h] if has_panel else []) + [content_row_h, caption_row_h]
    fig = plt.figure(figsize=(11.5, sum(row_heights)))
    gs = fig.add_gridspec(len(row_heights), 2, height_ratios=row_heights, hspace=0.55, wspace=0.25)

    r = 0
    if has_panel:
        ax_panel = fig.add_subplot(gs[r, :])
        ax_panel.imshow(Image.open(panel_path).convert("RGB"))
        ax_panel.axis("off")
        ax_panel.set_title(
            "Full reverse-generation trajectory for this instance (00 donor → 06 reference answer)",
            fontsize=10,
        )
        r += 1

    ax_img = fig.add_subplot(gs[r, 0])
    ax_img.imshow(Image.open(img_path).convert("RGB"))
    ax_img.axis("off")
    ax_img.set_title(f"Graded image\n{img_path.name}", fontsize=9)

    ax_grade = fig.add_subplot(gs[r, 1])
    plot_composite_grade(ax_grade, record)
    r += 1

    ax_caption = fig.add_subplot(gs[r, :])
    ax_caption.axis("off")
    ax_caption.text(
        0.0, 1.0, "\n".join(caption_lines), transform=ax_caption.transAxes,
        ha="left", va="top", fontsize=7.5, family="monospace",
    )

    fig.suptitle(
        f"Exercise {record['exercise']} | round {record['round']} | from_correct_{record['source_exercise']} | "
        f"{record['instance']} | score_ratio={record['score_ratio']:.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
