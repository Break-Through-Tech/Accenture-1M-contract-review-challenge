"""Run Risk Rules v1 on the annotated clause spans and write sample outputs (issue #31).

The input is the human-annotated spans in ``data/processed/annotations.jsonl``, so the
output shows what the rules do with a correctly detected clause, independent of the
classifier. It reads the validation contracts by default; the test split is not used.

    python scripts/run_risk_rules.py
    python scripts/run_risk_rules.py --reviewer-alias Licensee --reviewer-alias Customer

Writes to ``results/risk_rules/``:
    sample_outputs.csv   one row per contract and category (full evidence)
    SAMPLE_OUTPUTS.md    level counts per category and worked examples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from risk_rules import RULES, RuleConfig, score_clause

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "results" / "risk_rules"
LEVELS = ["High", "Medium", "Low"]
EXAMPLES_PER_LEVEL = 1
EVIDENCE_CHARS = 420


def load_spans(split: str) -> pd.DataFrame:
    """One row per contract and rule category that has at least one annotated span."""
    splits = pd.read_csv(PROCESSED_DIR / "splits.csv")
    keep = set(splits.loc[splits["split"] == split, "contract_id"])
    rows = []
    with (PROCESSED_DIR / "annotations.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["contract_id"] in keep and row["category"] in RULES and row["has_annotation"]:
                spans = [a["text"] for a in row["annotations"] if a.get("text")]
                rows.append({"contract_id": row["contract_id"], "category": row["category"], "spans": spans})
    return pd.DataFrame(rows)


def score_all(spans: pd.DataFrame, aliases: tuple[str, ...], config: RuleConfig) -> pd.DataFrame:
    out = []
    for row in spans.itertuples():
        result = score_clause(row.category, row.spans, aliases, config)
        out.append(
            {
                "contract_id": row.contract_id,
                "category": row.category,
                "n_spans": len(row.spans),
                "level": result.level,
                "rule_id": result.rule_id,
                "rationale": result.rationale,
                "assumptions": " ".join(result.assumptions),
                "span_levels": ",".join(result.span_levels),
                "facts": json.dumps(result.facts, sort_keys=True),
                "evidence": result.evidence,
            }
        )
    return pd.DataFrame(out)


def level_counts(scored: pd.DataFrame) -> pd.DataFrame:
    counts = scored.pivot_table(index="category", columns="level", values="contract_id", aggfunc="count", fill_value=0)
    counts = counts.reindex(index=list(RULES), columns=LEVELS, fill_value=0)
    counts["default_rule"] = scored.groupby("category")["rule_id"].apply(lambda s: s.str.endswith(".default").sum())
    return counts


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join([frame.index.name or "", *map(str, frame.columns)]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    body = ["| " + " | ".join([str(i), *(str(v) for v in row)]) + " |" for i, row in zip(frame.index, frame.to_numpy())]
    return "\n".join([header, rule, *body])


def write_report(scored: pd.DataFrame, split: str, aliases: tuple[str, ...], path: Path) -> None:
    alias_note = f"reviewer aliases: {', '.join(aliases)}" if aliases else "no reviewer aliases given (direction of one-sided terms is unknown)"
    lines = [
        "# Risk Rules v1: sample outputs",
        "",
        f"Rules applied to the annotated spans of the **{split}** contracts ({scored['contract_id'].nunique()} contracts, "
        f"{len(scored)} contract-category rows); {alias_note}. "
        "These are gold spans, so the table shows rule behaviour on correctly detected clauses, not end-to-end quality. "
        "No advisor labels exist yet; agreement with them is the job of the Risk Validation task (#34).",
        "",
        "## Level counts by category",
        "",
        "`default_rule` counts rows where no rule could read the clause and Medium is the fallback.",
        "",
        markdown_table(level_counts(scored)),
        "",
        "## Worked examples",
    ]
    for category in RULES:
        lines += ["", f"### {category}"]
        subset = scored[scored["category"] == category]
        for level in LEVELS:
            rows = subset[(subset["level"] == level) & ~subset["rule_id"].str.endswith(".default")].head(EXAMPLES_PER_LEVEL)
            for row in rows.itertuples():
                evidence = row.evidence.replace("\n", " ")
                evidence = evidence[:EVIDENCE_CHARS] + ("..." if len(evidence) > EVIDENCE_CHARS else "")
                lines += [
                    "",
                    f"**{level}** (`{row.rule_id}`, {row.contract_id})",
                    f"> {evidence}",
                    f"Rationale: {row.rationale}",
                ]
                if row.assumptions:
                    lines.append(f"Assumption: {row.assumptions}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--reviewer-alias", action="append", default=[], help="party name or role of the reviewer; repeatable")
    parser.add_argument("--no-assume-adverse", action="store_true", help="do not read unknown-direction terms against the reviewer")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    config = RuleConfig(assume_adverse_when_direction_unknown=not args.no_assume_adverse)
    aliases = tuple(args.reviewer_alias)
    scored = score_all(load_spans(args.split), aliases, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "sample_outputs.csv", index=False)
    write_report(scored, args.split, aliases, args.output_dir / "SAMPLE_OUTPUTS.md")
    print(level_counts(scored).to_string())
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
