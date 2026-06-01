import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "target_match_model.py"
SPEC = importlib.util.spec_from_file_location("target_match_model", MODULE_PATH)
target_match_model = importlib.util.module_from_spec(SPEC) if SPEC and SPEC.loader else None
if SPEC and SPEC.loader:
    SPEC.loader.exec_module(target_match_model)


class TargetMatchModelTests(unittest.TestCase):
    def write_fixture(self, root: Path):
        structures = root / "structures.smiles.tsv"
        interactions = root / "drug.target.interaction.tsv.gz"
        with structures.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["SMILES", "InChI", "InChIKey", "ID", "INN", "CAS_RN"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "SMILES": "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1",
                        "ID": "2001",
                        "INN": "oseltamivir",
                    },
                    {
                        "SMILES": "CC(=O)N[C@@H]1[C@@H](NC(N)=N)C=C(O[C@H]1[C@H](O)[C@H](O)CO)C(O)=O",
                        "ID": "2859",
                        "INN": "zanamivir",
                    },
                    {
                        "SMILES": "CNC(=O)c1ccc2c(c1)C(=NN2)Cc3ccccn3",
                        "ID": "5392",
                        "INN": "capmatinib",
                    },
                    {
                        "SMILES": "CC(=O)Oc1ccccc1C(=O)O",
                        "ID": "100",
                        "INN": "aspirin",
                    },
                ]
            )
        with gzip.open(interactions, "wt", encoding="utf-8", newline="") as handle:
            fields = [
                "DRUG_NAME",
                "STRUCT_ID",
                "TARGET_NAME",
                "TARGET_CLASS",
                "ACCESSION",
                "GENE",
                "SWISSPROT",
                "ACT_VALUE",
                "ACT_UNIT",
                "ACT_TYPE",
                "ACT_COMMENT",
                "ACT_SOURCE",
                "RELATION",
                "MOA",
                "MOA_SOURCE",
                "ACT_SOURCE_URL",
                "MOA_SOURCE_URL",
                "ACTION_TYPE",
                "TDL",
                "ORGANISM",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "DRUG_NAME": "oseltamivir",
                        "STRUCT_ID": "2001",
                        "TARGET_NAME": "Neuraminidase",
                        "TARGET_CLASS": "Enzyme",
                        "ACCESSION": "P03468",
                        "SWISSPROT": "NRAM_I34A1",
                        "ACT_TYPE": "IC50",
                        "ACTION_TYPE": "INHIBITOR",
                        "ORGANISM": "Influenza A virus H1N1",
                    },
                    {
                        "DRUG_NAME": "zanamivir",
                        "STRUCT_ID": "2859",
                        "TARGET_NAME": "Neuraminidase",
                        "TARGET_CLASS": "Enzyme",
                        "ACCESSION": "P03468",
                        "SWISSPROT": "NRAM_I34A1",
                        "ACT_TYPE": "IC50",
                        "ACTION_TYPE": "INHIBITOR",
                        "ORGANISM": "Influenza A virus H1N1",
                    },
                    {
                        "DRUG_NAME": "capmatinib",
                        "STRUCT_ID": "5392",
                        "TARGET_NAME": "MET proto-oncogene receptor tyrosine kinase",
                        "TARGET_CLASS": "Kinase",
                        "ACCESSION": "P08581",
                        "GENE": "MET",
                        "ACTION_TYPE": "INHIBITOR",
                        "ORGANISM": "Homo sapiens",
                    },
                    {
                        "DRUG_NAME": "aspirin",
                        "STRUCT_ID": "100",
                        "TARGET_NAME": "Prostaglandin G/H synthase 1",
                        "TARGET_CLASS": "Enzyme",
                        "ACCESSION": "P23219",
                        "GENE": "PTGS1",
                        "ACTION_TYPE": "INHIBITOR",
                        "ORGANISM": "Homo sapiens",
                    },
                ]
            )
        return structures, interactions

    def test_build_training_pairs_creates_positive_and_negative_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            structures, interactions = self.write_fixture(Path(tmp))
            dataset = target_match_model.build_target_match_dataset(
                structures,
                interactions,
                negative_ratio=1,
                seed=7,
            )

            self.assertEqual(dataset.summary["positive_pairs"], 4)
            self.assertEqual(dataset.summary["negative_pairs"], 4)
            self.assertIn("P03468", dataset.targets_by_key)
            self.assertIn("2001", dataset.structures_by_id)
            self.assertEqual(len(dataset.examples), 8)

    def test_train_and_predict_target_drug_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures, interactions = self.write_fixture(root)
            model_dir = root / "model"

            result = target_match_model.train_target_match_model(
                structures,
                interactions,
                model_dir,
                epochs=8,
                negative_ratio=1,
                hidden_dim=32,
                batch_size=4,
                seed=11,
            )
            prediction = target_match_model.predict_target_drug_matches(
                model_dir,
                target_query="Neuraminidase influenza",
                top=4,
            )

            self.assertTrue(Path(result["files"]["model"]).exists())
            self.assertTrue(Path(result["files"]["metadata"]).exists())
            self.assertGreaterEqual(result["dataset"]["positive_pairs"], 4)
            self.assertEqual(prediction["target_query"], "Neuraminidase influenza")
            self.assertEqual(len(prediction["rows"]), 4)
            self.assertIn("match_probability", prediction["rows"][0])
            self.assertIn("computational", prediction["boundary"].lower())
            ranked_names = [row["drug_name"] for row in prediction["rows"]]
            self.assertIn("oseltamivir", ranked_names)
            self.assertIn("zanamivir", ranked_names)

    def test_exact_target_accession_query_prioritizes_known_target_drugs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures, interactions = self.write_fixture(root)
            model_dir = root / "model"

            target_match_model.train_target_match_model(
                structures,
                interactions,
                model_dir,
                epochs=8,
                negative_ratio=1,
                hidden_dim=32,
                batch_size=4,
                seed=11,
            )
            prediction = target_match_model.predict_target_drug_matches(
                model_dir,
                target_query="P03468",
                top=2,
            )

            ranked_names = [row["drug_name"] for row in prediction["rows"]]
            self.assertEqual(set(ranked_names), {"oseltamivir", "zanamivir"})
            self.assertGreaterEqual(prediction["rows"][0]["known_target_similarity"], 0.9)

    def test_target_name_evidence_does_not_boost_unrelated_trigram_overlap(self):
        query_vec = target_match_model.target_text_features("Influenza A H1N1 neuraminidase NA P03468 pocket")
        unrelated_target = {
            "target_key": "Q96IV0",
            "target_name": "Peptide-N(4)-(N-acetyl-beta-glucosaminyl)asparagine amidase",
            "accession": "Q96IV0",
            "gene": "NGLY1",
            "swissprot": "",
            "organism": "Homo sapiens",
            "target_text": "Peptide-N(4)-(N-acetyl-beta-glucosaminyl)asparagine amidase Homo sapiens",
        }

        score = target_match_model._target_evidence_similarity(
            "Influenza A H1N1 neuraminidase NA P03468 pocket",
            query_vec,
            unrelated_target,
            256,
        )

        self.assertLess(score, 0.5)

    def test_evaluate_target_match_model_reports_topk_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures, interactions = self.write_fixture(root)
            model_dir = root / "model"

            target_match_model.train_target_match_model(
                structures,
                interactions,
                model_dir,
                epochs=8,
                negative_ratio=1,
                hidden_dim=32,
                batch_size=4,
                seed=11,
            )
            evaluation = target_match_model.evaluate_target_match_model(
                model_dir,
                max_targets=4,
                top_k=(1, 3),
            )

            self.assertEqual(evaluation["command"], "target-match-evaluate")
            self.assertGreaterEqual(evaluation["metrics"]["hit_at_3"], 0.75)
            self.assertGreater(evaluation["metrics"]["mean_reciprocal_rank"], 0)
            self.assertTrue(evaluation["rows"])

    def test_prediction_assets_precompute_evidence_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures, interactions = self.write_fixture(root)
            model_dir = root / "model"

            target_match_model.train_target_match_model(
                structures,
                interactions,
                model_dir,
                epochs=8,
                negative_ratio=1,
                hidden_dim=32,
                batch_size=4,
                seed=11,
            )
            _, metadata, checkpoint = target_match_model.load_target_match_model(model_dir)
            assets = target_match_model._prediction_assets(metadata, checkpoint)

            self.assertIn("evidence_by_structure", assets)
            self.assertIn("2001", assets["evidence_by_structure"])
            evidence = assets["evidence_by_structure"]["2001"][0]
            self.assertIn("target_vec", evidence)
            self.assertIn("accession_tokens", evidence)

    def test_exact_evidence_dominates_high_model_partial_evidence(self):
        exact_low_model = target_match_model.calibrated_match_probability(0.0, 1.0)
        partial_high_model = target_match_model.calibrated_match_probability(1.0, 0.95)

        self.assertGreater(exact_low_model, partial_high_model)

    def test_single_letter_tokens_do_not_create_exact_target_match(self):
        query_vec = target_match_model.target_text_features("Influenza A H1N1 neuraminidase P03468 pocket")
        unrelated_target = {
            "target_key": "P04439",
            "target_name": "HLA class I histocompatibility antigen, A-3 alpha chain",
            "accession": "P04439",
            "gene": "HLA-A",
            "swissprot": "",
            "organism": "Homo sapiens",
            "target_text": "HLA class I histocompatibility antigen, A-3 alpha chain P04439 HLA-A Homo sapiens",
        }

        score = target_match_model._target_evidence_similarity(
            "Influenza A H1N1 neuraminidase P03468 pocket",
            query_vec,
            unrelated_target,
            256,
        )

        self.assertLess(score, 0.5)

    def test_swissprot_species_suffix_does_not_create_exact_target_match(self):
        query = "A0PJK1 SLC5A10 SC5AA_HUMAN Sodium glucose cotransporter 5 Homo sapiens"
        query_vec = target_match_model.target_text_features(query)
        unrelated_target = {
            "target_key": "P16066",
            "target_name": "Atrial natriuretic peptide receptor 1",
            "accession": "P16066",
            "gene": "NPR1",
            "swissprot": "ANPRA_HUMAN",
            "organism": "Homo sapiens",
            "target_text": "Atrial natriuretic peptide receptor 1 P16066 NPR1 ANPRA_HUMAN Homo sapiens",
        }

        score = target_match_model._target_evidence_similarity(query, query_vec, unrelated_target, 256)

        self.assertLess(score, 0.5)


if __name__ == "__main__":
    unittest.main()
