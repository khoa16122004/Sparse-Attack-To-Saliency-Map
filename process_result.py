"""Aggregate per-sample evaluation.json files into CNN LaTeX tables."""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evaluate_latex_generate import generate_tables as tables
from evaluate_script import run_aggregate_faithfulness as faithfulness
from evaluate_script import run_score
from tqdm.auto import tqdm


MIN_SAMPLES = 91


def load_record(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    context = result["context"]
    record = {
        **copy.deepcopy(result["faithfulness"]),
        "schema_version": result["schema_version"],
        "status": result["status"],
        "sample_dir": str(path.parent.resolve()),
        "context": context,
    }
    faithfulness.validate_record(record, path)
    if str(context["seed"]) in ("", "default", "unknown", "None"):
        raise ValueError(f"{path}: explicit seed required")
    attack = result["attack"]
    if not isinstance(attack["success"], bool):
        raise ValueError(f"{path}: attack success must be a boolean")
    sro = attack["spearman"]
    if sro is None:
        if not attack.get("spearman_error"):
            raise ValueError(f"{path}: missing reason for undefined Spearman")
    elif isinstance(sro, bool) or not isinstance(sro, (int, float)) or not math.isfinite(sro) or not -1 <= sro <= 1:
        raise ValueError(f"{path}: invalid Spearman: {sro}")
    localization = result["localization"]
    for source in ("clean", "perturbation"):
        iou = localization[f"{source}_iou"]
        if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not math.isfinite(iou) or not 0 <= iou <= 1:
            raise ValueError(f"{path}: invalid {source} IoU: {iou}")
        record["maps"][source]["iou"] = iou
    record["localization_protocol"] = localization["protocol"]
    record["localization_provenance"] = {
        key: localization[key] for key in ("source_sha256", "gt_mask_sha256")
    }
    return record, attack


def aggregate(paths, debug_files=False):
    groups = defaultdict(lambda: defaultdict(list))
    for path in tqdm(paths, desc="Aggregating samples", unit="sample"):
        if debug_files:
            print(f"[READ] {path}", flush=True)
        record, attack = load_record(path)
        if debug_files:
            print(f"[LOADED] {path}", flush=True)
        context = record["context"]
        model = tables.canonical_model(context["model"])
        if model not in tables.MODELS["attack"][0]:
            continue
        approach = faithfulness.approach_name(context)
        config = faithfulness.configuration(record) | {
            "eps": context["eps"], "loss_type": context["loss_type"],
            "localization_protocol": record["localization_protocol"],
        }
        key = tables.config_key(config)
        cell = tables.cell_key(model, context["explain_method"], approach)
        groups[key][(cell, str(context["seed"]))].append((path, record, attack))
    if not groups:
        raise ValueError("No CNN evaluation.json records found")

    output = {}
    for config, runs in tqdm(groups.items(), desc="Aggregating configs", unit="config"):
        attack_cells = defaultdict(dict)
        eligible = []
        skipped = []
        for (cell, seed), entries in tqdm(
            sorted(runs.items()), desc="Aggregating runs", unit="run", leave=False,
        ):
            run_dirs = {path.parent.parent.parent.resolve() for path, _, _ in entries}
            if len(run_dirs) != 1:
                raise ValueError(f"Multiple run directories for {cell}, seed={seed}: {sorted(map(str, run_dirs))}")
            if len(entries) < MIN_SAMPLES:
                skipped.append({"model": cell[0], "explain_method": cell[1],
                                "approach": cell[2], "seed": seed,
                                "n_samples": len(entries), "run_dir": str(next(iter(run_dirs)))})
                continue
            sro = [attack["spearman"] for _, _, attack in entries if attack["spearman"] is not None]
            if not sro:
                raise ValueError(f"No valid Spearman values for {cell}, seed={seed}")
            asr = mean(attack["success"] for _, _, attack in entries)
            avg_sro = mean(sro)
            tables.add_seed(attack_cells, cell, seed, {
                "asr": asr, "sro": avg_sro,
                "h_score": run_score._h_score(asr, avg_sro),
            })
            if len(sro) != len(entries):
                tqdm.write(
                    f"[WARN] {cell}, seed={seed}: "
                    f"{len(entries) - len(sro)} undefined Spearman values",
                    file=sys.stderr,
                )
            eligible.extend(record for _, record, _ in entries)
        if eligible:
            faith_groups = tables.faithfulness_groups(eligible, include_iou=True)
            if set(faith_groups) != {config}:
                raise ValueError("Inconsistent faithfulness configuration")
            faith_cells = faith_groups[config]
        else:
            faith_cells = {}
        output[config] = (attack_cells, faith_cells, skipped)
    return output


def cnn_only(latex):
    marker = r"\textit{ViT models}"
    if marker not in latex:
        raise ValueError("ViT section missing from table template")
    latex = latex.split(marker, 1)[0].rstrip()
    latex = latex.removesuffix(r"\vspace{0.5cm}").rstrip()
    return latex.replace("across CNN and ViT models", "across CNN models") + "\n\\end{table*}\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True,
                        help="Roots containing per-sample evaluation.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("result_latexs") / "processed")
    parser.add_argument("--debug-files", action="store_true",
                        help="Log each evaluation file before and after loading and validation")
    args = parser.parse_args(argv)
    paths = set()
    for root in args.input_dirs:
        if not root.is_dir():
            raise NotADirectoryError(root)
        for directory, _, filenames in tqdm(
            os.walk(root), desc=f"Scanning {root.name}", unit="dir",
        ):
            if "evaluation.json" in filenames:
                paths.add((Path(directory) / "evaluation.json").resolve())
    if not paths:
        raise ValueError("No evaluation.json files found")
    tqdm.write(f"[SCAN] Found {len(paths)} unique evaluation files")
    for config, (attack_cells, faith_cells, skipped) in tqdm(
        sorted(aggregate(sorted(paths), debug_files=args.debug_files).items()),
        desc="Writing tables", unit="config",
    ):
        folder = args.output_dir / hashlib.sha256(config.encode()).hexdigest()[:12]
        folder.mkdir(parents=True, exist_ok=True)
        for kind, cells, filename in (
            ("attack", attack_cells, "performance.tex"),
            ("faithfulness", faith_cells, "saliency_quality.tex"),
        ):
            latex, report = tables.render_table(
                kind, cells, use_ttest=False, precision=3,
                include_iou=kind == "faithfulness", sd_format="subscript",
            )
            latex = cnn_only(latex).replace(
                "Each seed is averaged",
                "Only runs with more than 90 evaluated samples are included. Each seed is averaged",
            )
            report["configuration"] = json.loads(config)
            report["minimum_samples_per_run"] = MIN_SAMPLES
            report["skipped_runs"] = skipped
            (folder / filename).write_text(latex, encoding="utf-8")
            (folder / f"{kind}_statistics.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
        tqdm.write(f"[OK] {folder}: {len(skipped)} runs below {MIN_SAMPLES} samples")


if __name__ == "__main__":
    main()
