#!/usr/bin/env python3
"""Local DrugCentral target-drug matching model.

This module trains a lightweight neural matcher on local DrugCentral
drug-target relations. The output is a computational prioritization signal,
not evidence of efficacy, potency, safety, toxicity, dose, or clinical value.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - tests run with RDKit available in this workspace
    Chem = None
    Descriptors = None
    Lipinski = None
    rdFingerprintGenerator = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURES = ROOT / "data_lake" / "drugcentral" / "2021_09_01" / "structures.smiles.tsv"
DEFAULT_INTERACTIONS = ROOT / "data_lake" / "drugcentral" / "2021_09_01" / "drug.target.interaction.tsv.gz"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "target_match_drugcentral"
BOUNDARY = (
    "Computational target-drug matching only. This model does not prove biological activity, "
    "potency, efficacy, toxicity, safety, dosing, or clinical usefulness."
)


class TargetMatchDataset:
    def __init__(
        self,
        examples: List[Dict[str, object]],
        structures_by_id: Dict[str, Dict[str, str]],
        targets_by_key: Dict[str, Dict[str, str]],
        drug_target_index: Dict[str, List[str]],
        summary: Dict[str, object],
    ):
        self.examples = examples
        self.structures_by_id = structures_by_id
        self.targets_by_key = targets_by_key
        self.drug_target_index = drug_target_index
        self.summary = summary


class TargetMatchNet(nn.Module):
    def __init__(self, mol_dim: int, target_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mol_net = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.target_net = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mol_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        mol_h = self.mol_net(mol_x)
        target_h = self.target_net(target_x)
        joined = torch.cat([mol_h, target_h, torch.abs(mol_h - target_h), mol_h * target_h], dim=1)
        return self.classifier(joined).squeeze(1)


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _read_gzip_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _target_key(row: Dict[str, str]) -> str:
    for key in ["ACCESSION", "GENE", "SWISSPROT", "TARGET_NAME"]:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value.split("|")[0].strip()
    return ""


def _target_text(row: Dict[str, str]) -> str:
    parts = [
        row.get("TARGET_NAME", ""),
        row.get("TARGET_CLASS", ""),
        row.get("ACCESSION", ""),
        row.get("GENE", ""),
        row.get("SWISSPROT", ""),
        row.get("ACTION_TYPE", ""),
        row.get("ACT_TYPE", ""),
        row.get("ORGANISM", ""),
    ]
    return " ".join(str(part or "").strip() for part in parts if str(part or "").strip())


def _stable_hash(text: str) -> int:
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens: List[str] = []
    current = []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    compact = "".join(tokens)
    tokens.extend(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
    return [token for token in tokens if token]


def _word_tokens(text: str) -> List[str]:
    text = text.lower()
    tokens: List[str] = []
    current = []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return [token for token in tokens if token]


def _identifier_tokens(text: str) -> set:
    tokens = set()
    for raw in str(text or "").lower().replace("|", " ").split():
        cleaned = "".join(char if char.isalnum() else "_" for char in raw).strip("_")
        if len(cleaned) <= 1:
            continue
        tokens.add(cleaned)
        compact = "".join(char for char in cleaned if char.isalnum())
        if len(compact) > 1:
            tokens.add(compact)
    return tokens


def target_text_features(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokenize(text):
        idx = _stable_hash(token) % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def molecule_features(smiles: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim + 8, dtype=np.float32)
    if Chem is None or rdFingerprintGenerator is None:
        return vec
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return vec
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=dim).GetFingerprintAsNumPy(mol)
    vec[:dim] = np.asarray(fp, dtype=np.float32)
    if Descriptors is not None and Lipinski is not None:
        scalars = np.array(
            [
                min(float(Descriptors.MolWt(mol)) / 900.0, 2.0),
                min(max(float(Descriptors.MolLogP(mol)), -5.0), 10.0) / 10.0,
                min(float(Descriptors.TPSA(mol)) / 250.0, 2.0),
                min(float(Lipinski.NumHDonors(mol)) / 10.0, 2.0),
                min(float(Lipinski.NumHAcceptors(mol)) / 16.0, 2.0),
                min(float(Lipinski.NumRotatableBonds(mol)) / 20.0, 2.0),
                min(float(mol.GetNumHeavyAtoms()) / 100.0, 2.0),
                min(float(Lipinski.RingCount(mol)) / 12.0, 2.0),
            ],
            dtype=np.float32,
        )
        vec[dim:] = scalars
    return vec


def _load_structures(structures_path: Path) -> Dict[str, Dict[str, str]]:
    structures_by_id: Dict[str, Dict[str, str]] = {}
    for idx, row in enumerate(_read_tsv(structures_path), start=1):
        struct_id = str(row.get("ID", "") or idx).strip()
        smiles = str(row.get("SMILES", "") or "").strip()
        if not struct_id or not smiles:
            continue
        structures_by_id[struct_id] = {
            "struct_id": struct_id,
            "drug_name": str(row.get("INN", "") or f"DrugCentral structure {struct_id}").strip(),
            "smiles": smiles,
            "inchikey": str(row.get("InChIKey", "") or "").strip(),
            "cas_rn": str(row.get("CAS_RN", "") or "").strip(),
        }
    return structures_by_id


def build_target_match_dataset(
    structures_path: Path = DEFAULT_STRUCTURES,
    interactions_path: Path = DEFAULT_INTERACTIONS,
    negative_ratio: int = 1,
    max_positive_pairs: int = 0,
    seed: int = 13,
) -> TargetMatchDataset:
    structures_by_id = _load_structures(Path(structures_path))
    interactions = _read_gzip_tsv(Path(interactions_path))
    targets_by_key: Dict[str, Dict[str, str]] = {}
    positive_pairs = set()
    drug_target_index: Dict[str, List[str]] = {}

    for row in interactions:
        struct_id = str(row.get("STRUCT_ID", "") or "").strip()
        key = _target_key(row)
        if not struct_id or struct_id not in structures_by_id or not key:
            continue
        if key not in targets_by_key:
            targets_by_key[key] = {
                "target_key": key,
                "target_name": str(row.get("TARGET_NAME", "") or "").strip(),
                "target_class": str(row.get("TARGET_CLASS", "") or "").strip(),
                "accession": str(row.get("ACCESSION", "") or "").strip(),
                "gene": str(row.get("GENE", "") or "").strip(),
                "swissprot": str(row.get("SWISSPROT", "") or "").strip(),
                "organism": str(row.get("ORGANISM", "") or "").strip(),
                "action_type": str(row.get("ACTION_TYPE", "") or "").strip(),
                "act_type": str(row.get("ACT_TYPE", "") or "").strip(),
                "target_text": _target_text(row),
            }
        positive_pairs.add((struct_id, key))
        drug_target_index.setdefault(struct_id, [])
        if key not in drug_target_index[struct_id]:
            drug_target_index[struct_id].append(key)

    positives = sorted(positive_pairs)
    if max_positive_pairs and max_positive_pairs > 0:
        positives = positives[: int(max_positive_pairs)]

    rng = random.Random(seed)
    all_targets = sorted(targets_by_key)
    examples: List[Dict[str, object]] = [
        {"struct_id": struct_id, "target_key": target_key, "label": 1.0, "pair_source": "drugcentral_positive"}
        for struct_id, target_key in positives
    ]
    positive_by_struct: Dict[str, set] = {}
    for struct_id, target_key in positive_pairs:
        positive_by_struct.setdefault(struct_id, set()).add(target_key)

    for struct_id, target_key in positives:
        available = [candidate for candidate in all_targets if candidate not in positive_by_struct.get(struct_id, set())]
        if not available:
            continue
        for _ in range(max(0, int(negative_ratio))):
            negative_target = rng.choice(available)
            examples.append(
                {
                    "struct_id": struct_id,
                    "target_key": negative_target,
                    "label": 0.0,
                    "pair_source": "deterministic_negative_sample",
                }
            )

    rng.shuffle(examples)
    summary = {
        "positive_pairs": len(positives),
        "negative_pairs": len(examples) - len(positives),
        "examples": len(examples),
        "structures": len(structures_by_id),
        "targets": len(targets_by_key),
        "negative_ratio": int(negative_ratio),
        "boundary": BOUNDARY,
    }
    return TargetMatchDataset(examples, structures_by_id, targets_by_key, drug_target_index, summary)


def _features_for_examples(dataset: TargetMatchDataset, mol_dim: int, target_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol_cache: Dict[str, np.ndarray] = {}
    target_cache: Dict[str, np.ndarray] = {}
    mol_x, target_x, labels = [], [], []
    for example in dataset.examples:
        struct_id = str(example["struct_id"])
        target_key = str(example["target_key"])
        if struct_id not in mol_cache:
            mol_cache[struct_id] = molecule_features(dataset.structures_by_id[struct_id]["smiles"], mol_dim)
        if target_key not in target_cache:
            target_cache[target_key] = target_text_features(dataset.targets_by_key[target_key]["target_text"], target_dim)
        mol_x.append(mol_cache[struct_id])
        target_x.append(target_cache[target_key])
        labels.append(float(example["label"]))
    return np.vstack(mol_x), np.vstack(target_x), np.asarray(labels, dtype=np.float32)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"accuracy": 0.0, "roc_auc": 0.0}
    pred = (y_prob >= 0.5).astype(np.float32)
    accuracy = float((pred == y_true).mean())
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else 0.0
    except Exception:
        auc = 0.0
    return {"accuracy": round(accuracy, 4), "roc_auc": round(auc, 4)}


def train_target_match_model(
    structures_path: Path = DEFAULT_STRUCTURES,
    interactions_path: Path = DEFAULT_INTERACTIONS,
    model_dir: Path = DEFAULT_MODEL_DIR,
    epochs: int = 8,
    negative_ratio: int = 1,
    hidden_dim: int = 128,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    mol_dim: int = 512,
    target_dim: int = 256,
    max_positive_pairs: int = 0,
    seed: int = 13,
) -> Dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    dataset = build_target_match_dataset(
        Path(structures_path),
        Path(interactions_path),
        negative_ratio=negative_ratio,
        max_positive_pairs=max_positive_pairs,
        seed=seed,
    )
    mol_x, target_x, labels = _features_for_examples(dataset, mol_dim, target_dim)
    indices = np.arange(labels.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_idx = indices[:split]
    valid_idx = indices[split:] if split < len(indices) else indices[:0]

    model = TargetMatchNet(mol_x.shape[1], target_x.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_data = TensorDataset(
        torch.tensor(mol_x[train_idx], dtype=torch.float32),
        torch.tensor(target_x[train_idx], dtype=torch.float32),
        torch.tensor(labels[train_idx], dtype=torch.float32),
    )
    loader = DataLoader(train_data, batch_size=max(1, int(batch_size)), shuffle=True)
    history = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        total_loss = 0.0
        for batch_mol, batch_target, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_mol, batch_target)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_y.shape[0])
        history.append({"epoch": epoch, "train_loss": round(total_loss / max(1, len(train_idx)), 6)})

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(
            model(torch.tensor(mol_x[train_idx], dtype=torch.float32), torch.tensor(target_x[train_idx], dtype=torch.float32))
        ).numpy()
        valid_prob = (
            torch.sigmoid(
                model(torch.tensor(mol_x[valid_idx], dtype=torch.float32), torch.tensor(target_x[valid_idx], dtype=torch.float32))
            ).numpy()
            if len(valid_idx)
            else np.asarray([], dtype=np.float32)
        )
    metrics = {
        "train": _binary_metrics(labels[train_idx], train_prob),
        "validation": _binary_metrics(labels[valid_idx], valid_prob),
        "history": history,
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "target_match_model.pt"
    metadata_path = model_dir / "metadata.json"
    report_path = model_dir / "training_report.md"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mol_input_dim": int(mol_x.shape[1]),
            "target_input_dim": int(target_x.shape[1]),
            "mol_fp_dim": int(mol_dim),
            "target_dim": int(target_dim),
            "hidden_dim": int(hidden_dim),
            "feature_version": "drugcentral_morgan512_targethash256_v1",
        },
        model_path,
    )
    metadata = {
        "schema_version": "0.1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": dataset.summary,
        "training": {
            "epochs": int(epochs),
            "negative_ratio": int(negative_ratio),
            "hidden_dim": int(hidden_dim),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
        },
        "metrics": metrics,
        "structures": list(dataset.structures_by_id.values()),
        "targets_by_key": dataset.targets_by_key,
        "drug_target_index": dataset.drug_target_index,
        "boundary": BOUNDARY,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(render_training_report(metadata), encoding="utf-8")
    return {
        "command": "target-match-train",
        "model_dir": str(model_dir),
        "dataset": dataset.summary,
        "metrics": metrics,
        "files": {"model": str(model_path), "metadata": str(metadata_path), "report": str(report_path)},
        "boundary": BOUNDARY,
    }


def render_training_report(metadata: Dict[str, object]) -> str:
    dataset = metadata.get("dataset", {}) if isinstance(metadata.get("dataset"), dict) else {}
    metrics = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
    train = metrics.get("train", {}) if isinstance(metrics.get("train"), dict) else {}
    valid = metrics.get("validation", {}) if isinstance(metrics.get("validation"), dict) else {}
    return "\n".join(
        [
            "# Target-Drug Match Model Training Report",
            "",
            "## Dataset",
            "",
            f"- Positive pairs: {dataset.get('positive_pairs', 0)}",
            f"- Negative pairs: {dataset.get('negative_pairs', 0)}",
            f"- Structures: {dataset.get('structures', 0)}",
            f"- Targets: {dataset.get('targets', 0)}",
            "",
            "## Metrics",
            "",
            f"- Train accuracy: {train.get('accuracy', 0)}",
            f"- Train ROC-AUC: {train.get('roc_auc', 0)}",
            f"- Validation accuracy: {valid.get('accuracy', 0)}",
            f"- Validation ROC-AUC: {valid.get('roc_auc', 0)}",
            "",
            "## Boundary",
            "",
            f"- {BOUNDARY}",
            "",
        ]
    )


def load_target_match_model(model_dir: Path = DEFAULT_MODEL_DIR) -> Tuple[TargetMatchNet, Dict[str, object], Dict[str, object]]:
    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "target_match_model.pt", map_location="cpu", weights_only=False)
    model = TargetMatchNet(
        int(checkpoint["mol_input_dim"]),
        int(checkpoint["target_input_dim"]),
        int(checkpoint.get("hidden_dim", 128)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata, checkpoint


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _target_evidence_similarity(
    query_text: str,
    query_vec: np.ndarray,
    target: Dict[str, str],
    target_dim: int,
) -> float:
    return _target_evidence_similarity_from_record(
        query_text,
        query_vec,
        _target_evidence_record(target, target_dim),
    )


def _target_evidence_record(target: Dict[str, str], target_dim: int) -> Dict[str, object]:
    name_tokens = {token for token in _word_tokens(str(target.get("target_name", ""))) if len(token) > 2}
    organism_tokens = {token for token in _word_tokens(str(target.get("organism", ""))) if len(token) > 2}
    exact_tokens = set()
    accession_tokens = _identifier_tokens(str(target.get("accession", "") or target.get("target_key", "")))
    gene_tokens = _identifier_tokens(str(target.get("gene", "")))
    swissprot_tokens = _identifier_tokens(str(target.get("swissprot", "")))
    exact_tokens.update(accession_tokens)
    exact_tokens.update(gene_tokens)
    exact_tokens.update(swissprot_tokens)
    return {
        "target_key": str(target.get("target_key", "") or ""),
        "target_name": str(target.get("target_name", "") or ""),
        "organism": str(target.get("organism", "") or ""),
        "exact_tokens": exact_tokens,
        "accession_tokens": accession_tokens,
        "gene_tokens": gene_tokens,
        "swissprot_tokens": swissprot_tokens,
        "name_tokens": name_tokens,
        "organism_tokens": organism_tokens,
        "target_vec": target_text_features(str(target.get("target_text", "")), target_dim),
    }


def _target_evidence_similarity_from_record(
    query_text: str,
    query_vec: np.ndarray,
    record: Dict[str, object],
) -> float:
    query_tokens = set(_word_tokens(query_text))
    query_identifiers = _identifier_tokens(query_text)
    exact_scores = []
    exact_tokens = record.get("exact_tokens", set())
    if isinstance(exact_tokens, set) and exact_tokens.intersection(query_identifiers):
        exact_scores.append(1.0)
    name_tokens = record.get("name_tokens", set())
    if name_tokens:
        overlap = sum(1 for token in set(name_tokens) if token in query_tokens)
        if overlap == len(set(name_tokens)):
            exact_scores.append(0.95)
        elif overlap:
            exact_scores.append(min(0.90, 0.45 + 0.15 * overlap))
    organism_tokens = record.get("organism_tokens", set())
    if isinstance(organism_tokens, set) and organism_tokens.intersection(query_tokens):
        exact_scores.append(0.15)

    target_vec = record.get("target_vec", np.zeros_like(query_vec))
    cosine_score = _cosine(query_vec, target_vec if isinstance(target_vec, np.ndarray) else np.zeros_like(query_vec))
    return max([cosine_score] + exact_scores) if exact_scores else cosine_score


def _best_evidence_similarity(
    query_text: str,
    query_vec: np.ndarray,
    target_keys: Sequence[str],
    targets_by_key: Dict[str, Dict[str, str]],
    target_dim: int,
    target_records_by_key: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[float, str]:
    best_score = 0.0
    best_key = ""
    for key in target_keys:
        if target_records_by_key and str(key) in target_records_by_key:
            record = target_records_by_key[str(key)]
            score = _target_evidence_similarity_from_record(query_text, query_vec, record)
        else:
            target = targets_by_key.get(str(key), {})
            score = _target_evidence_similarity(query_text, query_vec, target, target_dim)
        if score > best_score:
            best_score = score
            best_key = str(key)
    return best_score, best_key


def calibrated_match_probability(model_probability: float, evidence_similarity: float) -> float:
    """Blend neural generalization with local evidence.

    DrugCentral contains explicit target-drug relations. When a user query is
    close to an observed target, that evidence should dominate ranking; when no
    local evidence matches, the neural score still supplies a weak screening
    prior.
    """
    model_probability = max(0.0, min(1.0, float(model_probability)))
    evidence_similarity = max(0.0, min(1.0, float(evidence_similarity)))
    if evidence_similarity >= 0.999:
        return 0.90 + 0.10 * model_probability
    if evidence_similarity >= 0.90:
        return 0.70 + 0.20 * ((evidence_similarity - 0.90) / 0.099) + 0.05 * model_probability
    if evidence_similarity >= 0.55:
        return 0.25 * model_probability + 0.75 * evidence_similarity
    if evidence_similarity > 0:
        return 0.55 * model_probability + 0.45 * evidence_similarity
    return 0.55 * model_probability


def _prediction_assets(metadata: Dict[str, object], checkpoint: Dict[str, object]) -> Dict[str, object]:
    mol_fp_dim = int(checkpoint.get("mol_fp_dim", 512))
    target_dim = int(checkpoint.get("target_dim", 256))
    structures = [structure for structure in metadata.get("structures", []) if isinstance(structure, dict)]
    targets_by_key = metadata.get("targets_by_key", {}) if isinstance(metadata.get("targets_by_key"), dict) else {}
    drug_target_index = metadata.get("drug_target_index", {}) if isinstance(metadata.get("drug_target_index"), dict) else {}
    target_records_by_key = {
        str(key): _target_evidence_record(target, target_dim)
        for key, target in targets_by_key.items()
        if isinstance(target, dict)
    }
    evidence_by_structure = {
        str(struct_id): [
            target_records_by_key[str(key)]
            for key in keys
            if str(key) in target_records_by_key
        ]
        for struct_id, keys in drug_target_index.items()
        if isinstance(keys, list)
    }
    if structures:
        mol_matrix = np.vstack([molecule_features(str(structure.get("smiles", "")), mol_fp_dim) for structure in structures])
    else:
        mol_matrix = np.zeros((0, int(checkpoint.get("mol_input_dim", mol_fp_dim + 8))), dtype=np.float32)
    return {
        "mol_fp_dim": mol_fp_dim,
        "target_dim": target_dim,
        "structures": structures,
        "mol_tensor": torch.tensor(mol_matrix, dtype=torch.float32),
        "targets_by_key": targets_by_key,
        "drug_target_index": drug_target_index,
        "target_records_by_key": target_records_by_key,
        "evidence_by_structure": evidence_by_structure,
    }


def _rank_target_drug_matches_loaded(
    model: TargetMatchNet,
    metadata: Dict[str, object],
    checkpoint: Dict[str, object],
    target_query: str,
    top: int = 25,
    assets: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    assets = assets or _prediction_assets(metadata, checkpoint)
    target_dim = int(assets["target_dim"])
    query_vec = target_text_features(target_query, target_dim)
    structures = assets["structures"]
    mol_tensor = assets["mol_tensor"]
    targets_by_key = assets["targets_by_key"]
    drug_target_index = assets["drug_target_index"]
    target_records_by_key = assets.get("target_records_by_key", {})
    if not structures:
        return []

    rows = []
    with torch.no_grad():
        query_tensor = torch.tensor(np.repeat(query_vec.reshape(1, -1), int(mol_tensor.shape[0]), axis=0), dtype=torch.float32)
        model_probabilities = torch.sigmoid(model(mol_tensor, query_tensor)).numpy()
        for structure, model_probability_raw in zip(structures, model_probabilities):
            model_probability = float(model_probability_raw)
            evidence_similarity, evidence_target_key = _best_evidence_similarity(
                target_query,
                query_vec,
                drug_target_index.get(str(structure.get("struct_id", "")), []),
                targets_by_key,
                target_dim,
                target_records_by_key=target_records_by_key if isinstance(target_records_by_key, dict) else None,
            )
            final_score = calibrated_match_probability(model_probability, evidence_similarity)
            matched_target = targets_by_key.get(evidence_target_key, {})
            rows.append(
                {
                    "rank": 0,
                    "struct_id": structure.get("struct_id", ""),
                    "drug_name": structure.get("drug_name", ""),
                    "smiles": structure.get("smiles", ""),
                    "match_probability": round(final_score, 6),
                    "model_probability": round(model_probability, 6),
                    "known_target_similarity": round(evidence_similarity, 6),
                    "evidence_target_key": evidence_target_key,
                    "evidence_target_name": matched_target.get("target_name", ""),
                    "evidence_organism": matched_target.get("organism", ""),
                    "confidence": confidence_band(final_score, evidence_similarity),
                    "source": "DrugCentral relation-trained neural matcher",
                    "boundary": BOUNDARY,
                }
            )
    rows.sort(key=lambda row: float(row["match_probability"]), reverse=True)
    selected = rows[: max(1, int(top))]
    for idx, row in enumerate(selected, start=1):
        row["rank"] = idx
    return selected


def predict_target_drug_matches(
    model_dir: Path = DEFAULT_MODEL_DIR,
    target_query: str = "",
    top: int = 25,
    output_csv: Optional[Path] = None,
) -> Dict[str, object]:
    if not str(target_query or "").strip():
        raise ValueError("target_query is required")
    model, metadata, checkpoint = load_target_match_model(Path(model_dir))
    selected = _rank_target_drug_matches_loaded(model, metadata, checkpoint, target_query, top)
    payload = {
        "command": "target-match-predict",
        "target_query": target_query,
        "model_dir": str(model_dir),
        "count": len(selected),
        "rows": selected,
        "boundary": BOUNDARY,
    }
    if output_csv:
        write_prediction_csv(selected, Path(output_csv))
        payload["output_csv"] = str(output_csv)
    return payload


def _target_query_from_metadata(target: Dict[str, str]) -> str:
    parts = [
        target.get("accession", ""),
        target.get("gene", ""),
        target.get("swissprot", ""),
        target.get("target_name", ""),
        target.get("target_class", ""),
        target.get("organism", ""),
    ]
    return " ".join(str(part or "").strip() for part in parts if str(part or "").strip())


def evaluate_target_match_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    max_targets: int = 100,
    min_positive_drugs: int = 1,
    top_k: Sequence[int] = (1, 3, 5, 10),
    output_json: Optional[Path] = None,
) -> Dict[str, object]:
    model, metadata, checkpoint = load_target_match_model(Path(model_dir))
    targets_by_key = metadata.get("targets_by_key", {}) if isinstance(metadata.get("targets_by_key"), dict) else {}
    drug_target_index = metadata.get("drug_target_index", {}) if isinstance(metadata.get("drug_target_index"), dict) else {}
    target_to_structs: Dict[str, set] = {}
    for struct_id, keys in drug_target_index.items():
        for key in keys if isinstance(keys, list) else []:
            target_to_structs.setdefault(str(key), set()).add(str(struct_id))

    target_items = [
        (key, target)
        for key, target in sorted(targets_by_key.items())
        if isinstance(target, dict) and len(target_to_structs.get(str(key), set())) >= int(min_positive_drugs)
    ]
    if max_targets and int(max_targets) > 0:
        target_items = target_items[: int(max_targets)]
    clean_top_k = sorted({max(1, int(k)) for k in top_k})
    max_k = max(clean_top_k) if clean_top_k else 10
    hit_counts = {k: 0 for k in clean_top_k}
    reciprocal_ranks: List[float] = []
    rows: List[Dict[str, object]] = []
    assets = _prediction_assets(metadata, checkpoint)

    for key, target in target_items:
        positives = target_to_structs.get(str(key), set())
        query = _target_query_from_metadata(target)
        predictions = _rank_target_drug_matches_loaded(model, metadata, checkpoint, query, max_k, assets=assets)
        ranked_ids = [str(row.get("struct_id", "")) for row in predictions]
        first_hit_rank = 0
        for idx, struct_id in enumerate(ranked_ids, start=1):
            if struct_id in positives:
                first_hit_rank = idx
                break
        for k in clean_top_k:
            if any(struct_id in positives for struct_id in ranked_ids[:k]):
                hit_counts[k] += 1
        reciprocal_ranks.append((1.0 / first_hit_rank) if first_hit_rank else 0.0)
        rows.append(
            {
                "target_key": key,
                "target_name": target.get("target_name", ""),
                "query": query,
                "positive_drugs": len(positives),
                "first_hit_rank": first_hit_rank,
                "top_drug": predictions[0].get("drug_name", "") if predictions else "",
                "top_score": predictions[0].get("match_probability", 0) if predictions else 0,
            }
        )

    evaluated = len(target_items)
    metrics = {f"hit_at_{k}": round(hit_counts[k] / evaluated, 4) if evaluated else 0.0 for k in clean_top_k}
    metrics["mean_reciprocal_rank"] = round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0
    payload = {
        "command": "target-match-evaluate",
        "model_dir": str(model_dir),
        "evaluated_targets": evaluated,
        "min_positive_drugs": int(min_positive_drugs),
        "top_k": clean_top_k,
        "metrics": metrics,
        "rows": rows,
        "boundary": BOUNDARY,
        "note": "Top-K retrieval is measured against local DrugCentral known target-drug relations; it is not biological validation.",
    }
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        payload["output_json"] = str(output_json)
    return payload


def confidence_band(score: float, evidence_similarity: float) -> str:
    if evidence_similarity >= 0.70 and score >= 0.65:
        return "high_screening_prior"
    if score >= 0.55:
        return "medium_screening_prior"
    return "low_screening_prior"


def write_prediction_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "struct_id",
        "drug_name",
        "smiles",
        "match_probability",
        "model_probability",
        "known_target_similarity",
        "evidence_target_key",
        "evidence_target_name",
        "evidence_organism",
        "confidence",
        "source",
        "boundary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run local target-drug matching model.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train model from local DrugCentral structures and interactions.")
    train.add_argument("--structures", default=str(DEFAULT_STRUCTURES))
    train.add_argument("--interactions", default=str(DEFAULT_INTERACTIONS))
    train.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--negative-ratio", type=int, default=1)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--max-positive-pairs", type=int, default=0)
    train.add_argument("--seed", type=int, default=13)

    predict = sub.add_parser("predict", help="Rank known drugs for a target description.")
    predict.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    predict.add_argument("--target", required=True, help="Target description, name, gene, accession, disease context, or organism.")
    predict.add_argument("--top", type=int, default=25)
    predict.add_argument("--output-csv", default="")

    evaluate = sub.add_parser("evaluate", help="Evaluate Top-K retrieval against local DrugCentral relations.")
    evaluate.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    evaluate.add_argument("--max-targets", type=int, default=100)
    evaluate.add_argument("--min-positive-drugs", type=int, default=1)
    evaluate.add_argument("--top-k", default="1,3,5,10")
    evaluate.add_argument("--output-json", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        result = train_target_match_model(
            Path(args.structures),
            Path(args.interactions),
            Path(args.model_dir),
            epochs=args.epochs,
            negative_ratio=args.negative_ratio,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_positive_pairs=args.max_positive_pairs,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "predict":
        result = predict_target_drug_matches(
            Path(args.model_dir),
            target_query=args.target,
            top=args.top,
            output_csv=Path(args.output_csv) if args.output_csv else None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        top_k = [int(part.strip()) for part in str(args.top_k).split(",") if part.strip()]
        result = evaluate_target_match_model(
            Path(args.model_dir),
            max_targets=args.max_targets,
            min_positive_drugs=args.min_positive_drugs,
            top_k=top_k,
            output_json=Path(args.output_json) if args.output_json else None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
