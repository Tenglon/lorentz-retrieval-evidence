#!/usr/bin/env python3
"""Validate frozen evidence and generate all LaTeX fragments deterministically."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "slides" / "generated"
T_CRIT_N5 = 2.7764451051977987


def rows(name: str):
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate() -> None:
    dimensions = rows("dev_dimension.csv")
    assert len(dimensions) == 8
    for row in dimensions:
        values = [float(value) for value in row["seed_values"].split(";")]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / 4
        radius = T_CRIT_N5 * math.sqrt(variance) / math.sqrt(5)
        assert abs(mean - float(row["mean"])) < 5e-7
        # CSV summaries are intentionally rounded to six decimals.
        assert abs(mean - radius - float(row["ci_low"])) < 5e-6
        assert abs(mean + radius - float(row["ci_high"])) < 5e-6
    full = rows("full_corpus.csv")
    assert len(full) == 8
    for row in full:
        assert abs(
            float(row["lorentz"])
            - float(row["euclidean"])
            - float(row["delta"])
        ) < 2e-6
    bounded = rows("bounded_screen.csv")
    for row in bounded:
        assert int(row["wins"]) + int(row["ties"]) + int(row["losses"]) == 97
        assert (
            int(row["positive_ci"])
            + int(row["negative_ci"])
            + int(row["crosses_zero"])
            == 97
        )


def write_dimension_plot() -> None:
    grouped = {"Lorentz": [], "Euclidean": []}
    for row in rows("dev_dimension.csv"):
        grouped[row["geometry"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["dimension"]))
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[width=0.88\linewidth,height=0.48\textheight,",
        r"xlabel={embedding dimension},ylabel={development macro MRR},",
        r"xmode=log,log basis x=2,xtick={32,64,128,256},",
        r"ymin=0.72,ymax=0.86,grid=major,legend pos=south east]",
    ]
    styles = {"Lorentz": "blue,mark=*", "Euclidean": "orange!85!black,mark=square*"}
    for geometry in ("Euclidean", "Lorentz"):
        coordinates = " ".join(
            f"({row['dimension']},{row['mean']}) +- (0,{(float(row['ci_high'])-float(row['ci_low']))/2:.6f})"
            for row in grouped[geometry]
        )
        lines += [
            rf"\addplot+[{styles[geometry]},thick,error bars/.cd,y dir=both,y explicit] coordinates {{{coordinates}}};",
            rf"\addlegendentry{{{geometry}}}",
        ]
    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    (OUT / "dimension_plot.tex").write_text("\n".join(lines) + "\n")


def write_full_task_plot() -> None:
    values = [row for row in rows("full_corpus.csv") if row["dimension"] == "128"]
    symbolic = ",".join("Nano" if row["task"] == "NanoSciFact" else row["task"].split()[0] for row in values)
    coordinates = " ".join(
        f"({row['delta']},{'Nano' if row['task']=='NanoSciFact' else row['task'].split()[0]})"
        f" += ({float(row['ci_high'])-float(row['delta']):.6f},0)"
        f" -= ({float(row['delta'])-float(row['ci_low']):.6f},0)"
        for row in values
    )
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.92\linewidth,height=0.42\textheight,
xlabel={{H $-$ E on official task metric}},symbolic y coords={{{symbolic}}},
ytick=data,xmin=-0.04,xmax=0.105,grid=major,axis y line*=left]
\addplot+[only marks,blue,mark=*,mark size=2.4pt,
error bars/.cd,x dir=both,x explicit] coordinates {{{coordinates}}};
\draw[dashed,gray] (axis cs:0,{symbolic.split(',')[0]}) -- (axis cs:0,{symbolic.split(',')[-1]});
\end{{axis}}
\end{{tikzpicture}}
"""
    (OUT / "full_task_plot.tex").write_text(text)


def write_tables() -> None:
    bounded = rows("bounded_screen.csv")
    full = [row for row in rows("full_corpus.csv") if row["dimension"] == "128"]
    b_lines = [
        r"\begin{tabular}{@{}rrrrrr@{}}",
        r"\toprule",
        r"Dim & $E$ & $H$ & $H-E$ [95\% CI] & W/T/L \\",
        r"\midrule",
    ]
    for row in bounded:
        b_lines.append(
            f"{row['dimension']} & {float(row['e_macro']):.4f} & {float(row['h_macro']):.4f} & "
            f"{float(row['delta']):+.4f} [{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] & "
            f"{row['wins']}/{row['ties']}/{row['losses']} \\\\"
        )
    b_lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "bounded_table.tex").write_text("\n".join(b_lines) + "\n")
    f_lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Task & $E$ & $H$ & $H-E$ \\", r"\midrule"]
    for row in full:
        task = row["task"].replace("MS-COCO text-to-image", "MS-COCO T$\to$I")
        f_lines.append(
            f"{task} & {float(row['euclidean']):.4f} & {float(row['lorentz']):.4f} & {float(row['delta']):+.4f} \\\\"
        )
    f_lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "full_table.tex").write_text("\n".join(f_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate only")
    args = parser.parse_args()
    validate()
    if args.check:
        print("evidence validation: OK")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    write_dimension_plot()
    write_full_task_plot()
    write_tables()
    print(f"generated assets in {OUT}")


if __name__ == "__main__":
    main()
