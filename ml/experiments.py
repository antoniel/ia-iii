from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.validate import ValidateResult

PRIMARY_METRIC = "roc_auc"
DEFAULT_LOG_PATH = Path("experiments/experiments.jsonl")


@dataclass
class ExperimentRecord:
    id: str
    timestamp: str
    tag: str
    notes: str
    primary_metric: str
    metrics: dict[str, float]
    delta_vs_best: dict[str, float | None]
    is_best: bool
    config: dict[str, Any]
    top_importances: dict[str, float]
    n_train: int
    n_test: int
    n_features: int
    model_path: str


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _next_id(path: Path) -> str:
    records = _read_records(path)
    return f"exp_{len(records) + 1:03d}"


def _best_record(
    records: list[dict[str, Any]],
    metric: str = PRIMARY_METRIC,
) -> dict[str, Any] | None:
    if not records:
        return None
    return max(records, key=lambda record: record["metrics"][metric])


def _delta_vs_best(
    metrics: dict[str, float],
    best: dict[str, Any] | None,
) -> dict[str, float | None]:
    if best is None:
        return {name: None for name in metrics}
    return {name: round(metrics[name] - best["metrics"][name], 4) for name in metrics}


def _serialize_run_config(config: Any) -> dict[str, Any]:
    raw = asdict(config)
    serialized: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {
            "experiments_path",
            "log_experiments",
            "experiment_tag",
            "experiment_notes",
        }:
            continue
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif key == "feature_columns":
            serialized[key] = [
                col.value if hasattr(col, "value") else str(col) for col in value
            ]
        else:
            serialized[key] = value
    return serialized


def log_experiment(
    run_config: Any,
    result: ValidateResult,
    *,
    path: Path = DEFAULT_LOG_PATH,
    tag: str = "",
    notes: str = "",
    n_train: int = 0,
    n_test: int = 0,
    n_features: int = 0,
    metric: str = PRIMARY_METRIC,
) -> ExperimentRecord:
    existing = _read_records(path)
    previous_best = _best_record(existing, metric)
    delta = _delta_vs_best(result.metrics, previous_best)
    is_best = (
        previous_best is None
        or result.metrics[metric] > previous_best["metrics"][metric]
    )

    record = ExperimentRecord(
        id=_next_id(path),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        tag=tag,
        notes=notes,
        primary_metric=metric,
        metrics={name: round(value, 4) for name, value in result.metrics.items()},
        delta_vs_best=delta,
        is_best=is_best,
        config=_serialize_run_config(run_config),
        top_importances={
            name: round(score, 4) for name, score in result.importances.head(5).items()
        },
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        model_path=str(run_config.model_path),
    )

    if is_best and previous_best is not None:
        for entry in existing:
            entry["is_best"] = False

    existing.append(asdict(record))
    _write_records(path, existing)
    return record


def annotate(path: Path, experiment_id: str, notes: str) -> ExperimentRecord:
    records = _read_records(path)
    for record in records:
        if record["id"] == experiment_id:
            record["notes"] = notes
            _write_records(path, records)
            return ExperimentRecord(**record)
    raise ValueError(f"Experiment not found: {experiment_id}")


def format_log_summary(record: ExperimentRecord, path: Path) -> str:
    metric = record.primary_metric
    value = record.metrics[metric]
    lines = [
        f"Experiment {record.id} logged → {path}",
        f"   {metric}={value:.3f}",
    ]
    if record.tag:
        lines.append(f"   tag: {record.tag}")
    if record.is_best:
        delta = record.delta_vs_best[metric]
        if delta is None:
            lines.append("   ★ first run — baseline")
        else:
            lines.append(f"   ★ new best ({metric} {delta:+.3f} vs previous best)")
    elif record.delta_vs_best[metric] is not None:
        lines.append(f"   delta vs best: {metric} {record.delta_vs_best[metric]:+.3f}")
    if not record.notes:
        lines.append(f'   add notes: uv run ia-iii-exp note {record.id} "..."')
    return "\n".join(lines)


def _cmd_note(args: argparse.Namespace) -> None:
    record = annotate(args.path, args.id, args.text)
    print(f"Updated {record.id}: {record.notes}")


def _cmd_list(args: argparse.Namespace) -> None:
    records = _read_records(args.path)
    if not records:
        print(f"No experiments in {args.path}")
        return
    for record in records:
        metric = record["primary_metric"]
        best_mark = " ★" if record.get("is_best") else ""
        tag = f" [{record['tag']}]" if record.get("tag") else ""
        notes = f" — {record['notes']}" if record.get("notes") else ""
        print(
            f"{record['id']}{best_mark}{tag}: "
            f"{metric}={record['metrics'][metric]:.3f}{notes}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment log for ia-iii")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to experiments JSONL",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    note_parser = sub.add_parser("note", help="Add or update experiment notes")
    note_parser.add_argument("id", help="Experiment id, e.g. exp_001")
    note_parser.add_argument("text", help="What changed in this run")
    note_parser.set_defaults(func=_cmd_note)

    list_parser = sub.add_parser("list", help="List logged experiments")
    list_parser.set_defaults(func=_cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
