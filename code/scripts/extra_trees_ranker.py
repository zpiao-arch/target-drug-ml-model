#!/usr/bin/env python3
"""Standalone ExtraTrees target-drug ranking script.

Run from the package root:

python3 code/scripts/extra_trees_ranker.py \
  --target-query "Influenza A neuraminidase oseltamivir zanamivir P03468" \
  --top 10 \
  --output experiments/influenza_na_rank_from_package.csv

The output is a computational screening priority, not biological or clinical
validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PACKAGE_ROOT / "code" / "ai_mol_loop"
sys.path.insert(0, str(CODE_ROOT))

import target_match_model as tmm  # noqa: E402


BOUNDARY = (
    "Computational target-drug matching only. This ranking does not prove "
    "biological activity, potency, efficacy, toxicity, safety, dosing, "
    "clinical usefulness, or suitability for patient care."
)


def read_approval_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    static_dir = path / "static"
    files = {
        "FDA": static_dir / "FDA_Approved.csv",
        "EMA": static_dir / "EMA_Approved.csv",
        "PMDA": static_dir / "PMDA_Approved.csv",
    }
    for region, file_path in files.items():
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                struct_id = str(row[0] or "").strip()
                if not struct_id or struct_id.lower() in {"id", "struct_id"}:
                    continue
                entry = rows.setdefault(
                    struct_id,
                    {
                        "approval_fda": False,
                        "approval_ema": False,
                        "approval_pmda": False,
                        "approval_regions": [],
                    },
                )
                entry[f"approval_{region.lower()}"] = True
                entry["approval_regions"].append(region)
    for entry in rows.values():
        regions = list(dict.fromkeys(entry.get("approval_regions", [])))
        entry["approval_region_count"] = len(regions)
        entry["approval_regions"] = ";".join(regions)
        if len(regions) >= 3:
            entry["approval_tier"] = "fda_ema_pmda"
        elif len(regions) >= 2:
            entry["approval_tier"] = "multi_region"
        elif "FDA" in regions:
            entry["approval_tier"] = "fda"
        elif "EMA" in regions:
            entry["approval_tier"] = "ema"
        elif "PMDA" in regions:
            entry["approval_tier"] = "pmda"
        else:
            entry["approval_tier"] = "unknown"
    return rows


def rank_drugs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    model_path = Path(args.model)
    structures_path = Path(args.structures)
    interactions_path = Path(args.interactions)
    data_root = structures_path.parents[1]
    model = joblib.load(model_path)

    dataset = tmm.build_target_match_dataset(
        structures_path=structures_path,
        interactions_path=interactions_path,
        negative_ratio=0,
        seed=int(args.seed),
    )
    structures = list(dataset.structures_by_id.values())
    mol_matrix = np.vstack(
        [tmm.molecule_features(str(structure.get("smiles", "")), 512) for structure in structures]
    ).astype(np.float32)
    target_records = {
        str(key): tmm._target_evidence_record(target, 256)
        for key, target in dataset.targets_by_key.items()
        if isinstance(target, dict)
    }
    approvals = read_approval_rows(data_root)

    query = str(args.target_query or "").strip()
    if not query:
        raise SystemExit("--target-query is required")
    query_vec = tmm.target_text_features(query, 256)
    target_matrix = np.repeat(query_vec.reshape(1, -1), len(structures), axis=0)
    feature_matrix = np.hstack([mol_matrix, target_matrix]).astype(np.float32)
    model_probabilities = model.predict_proba(feature_matrix)[:, 1]

    rows: List[Dict[str, Any]] = []
    for structure, model_probability in zip(structures, model_probabilities):
        struct_id = str(structure.get("struct_id", ""))
        evidence_similarity, evidence_target_key = tmm._best_evidence_similarity(
            query,
            query_vec,
            dataset.drug_target_index.get(struct_id, []),
            dataset.targets_by_key,
            256,
            target_records_by_key=target_records,
        )
        final_score = tmm.calibrated_match_probability(float(model_probability), evidence_similarity)
        matched_target = dataset.targets_by_key.get(evidence_target_key, {})
        approval = approvals.get(
            struct_id,
            {
                "approval_fda": False,
                "approval_ema": False,
                "approval_pmda": False,
                "approval_regions": "",
                "approval_region_count": 0,
                "approval_tier": "unknown",
            },
        )
        rows.append(
            {
                "rank": 0,
                "struct_id": struct_id,
                "drug_name": str(structure.get("drug_name", "")),
                "smiles": str(structure.get("smiles", "")),
                "score": round(float(final_score), 6),
                "model_probability": round(float(model_probability), 6),
                "known_target_similarity": round(float(evidence_similarity), 6),
                "evidence_target_key": evidence_target_key,
                "evidence_target": str(matched_target.get("target_name", "")),
                "evidence_organism": str(matched_target.get("organism", "")),
                "confidence": tmm.confidence_band(float(final_score), float(evidence_similarity)),
                "approval_fda": approval.get("approval_fda", False),
                "approval_ema": approval.get("approval_ema", False),
                "approval_pmda": approval.get("approval_pmda", False),
                "approval_regions": approval.get("approval_regions", ""),
                "approval_region_count": approval.get("approval_region_count", 0),
                "approval_tier": approval.get("approval_tier", "unknown"),
                "boundary": BOUNDARY,
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = rows[: max(1, int(args.top))]
    for idx, row in enumerate(selected, start=1):
        row["rank"] = idx
    return selected


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "struct_id",
        "drug_name",
        "smiles",
        "score",
        "model_probability",
        "known_target_similarity",
        "evidence_target_key",
        "evidence_target",
        "evidence_organism",
        "confidence",
        "approval_fda",
        "approval_ema",
        "approval_pmda",
        "approval_regions",
        "approval_region_count",
        "approval_tier",
        "boundary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-query", required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--model",
        default=str(PACKAGE_ROOT / "models" / "target_match_drugcentral" / "extra_trees" / "target_match_extra_trees.joblib"),
    )
    parser.add_argument(
        "--structures",
        default=str(PACKAGE_ROOT / "data" / "drugcentral" / "2021_09_01" / "structures.smiles.tsv"),
    )
    parser.add_argument(
        "--interactions",
        default=str(PACKAGE_ROOT / "data" / "drugcentral" / "2021_09_01" / "drug.target.interaction.tsv.gz"),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = rank_drugs(args)
    if args.output:
        write_csv(rows, Path(args.output))
    print(json.dumps({"count": len(rows), "rows": rows, "boundary": BOUNDARY}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
