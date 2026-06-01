import csv
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import server


class MlRankApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.projects_root = Path(self.tmp.name) / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.original_projects_root = server.PROJECTS_ROOT
        server.PROJECTS_ROOT = self.projects_root
        self.client = TestClient(server.app)

    def tearDown(self):
        server.PROJECTS_ROOT = self.original_projects_root
        self.tmp.cleanup()

    def test_one_click_ml_rank_creates_drugcentral_ranking_and_agent_context(self):
        create = self.client.post("/api/projects", json={"name": "ml_rank_demo"})
        self.assertEqual(create.status_code, 200, create.text)

        response = self.client.post(
            "/api/projects/ml_rank_demo/ml-rank/run",
            json={
                "round": 1,
                "disease": "甲流",
                "target_query": "Influenza A neuraminidase oseltamivir zanamivir peramivir P03468",
                "top": 10,
                "approval_filter": "all",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        self.assertEqual(payload["command"], "ml-rank-run")
        self.assertEqual(payload["model"]["type"], "sklearn.ensemble.ExtraTreesClassifier")
        self.assertEqual(payload["count"], 10)
        self.assertIn("Computational", payload["boundary"])
        self.assertTrue(payload["files"]["ranking_csv"].endswith("round_1_ml_drug_rankings.csv"))
        self.assertTrue(payload["files"]["agent_context"].endswith("round_1_ml_agent_context.md"))

        rows = payload["rows"]
        top_names = {str(row["drug_name"]).lower() for row in rows[:5]}
        self.assertTrue({"zanamivir", "oseltamivir", "peramivir"} & top_names)
        for field in [
            "rank",
            "struct_id",
            "drug_name",
            "smiles",
            "score",
            "model_probability",
            "known_target_similarity",
            "evidence_target",
            "source_confidence",
            "approval_tier",
        ]:
            self.assertIn(field, rows[0])

        ranking_path = Path(payload["files"]["ranking_csv"])
        self.assertTrue(ranking_path.exists())
        with ranking_path.open("r", encoding="utf-8-sig", newline="") as handle:
            saved_rows = list(csv.DictReader(handle))
        self.assertEqual(len(saved_rows), 10)
        self.assertEqual(saved_rows[0]["rank"], "1")

    def test_agent_analysis_reads_saved_ml_ranking(self):
        create = self.client.post("/api/projects", json={"name": "ml_agent_demo"})
        self.assertEqual(create.status_code, 200, create.text)
        run = self.client.post(
            "/api/projects/ml_agent_demo/ml-rank/run",
            json={
                "round": 1,
                "disease": "甲流",
                "target_query": "Influenza A neuraminidase oseltamivir zanamivir peramivir P03468",
                "top": 8,
            },
        )
        self.assertEqual(run.status_code, 200, run.text)

        response = self.client.post(
            "/api/projects/ml_agent_demo/ml-rank/agent-analysis",
            json={"round": 1, "top": 5},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        self.assertEqual(payload["command"], "ml-rank-agent-analysis")
        self.assertEqual(payload["source"], "saved_ml_ranking")
        self.assertGreaterEqual(len(payload["top_candidates"]), 3)
        self.assertIn("ExtraTrees", payload["analysis"])
        self.assertIn("计算筛选", payload["analysis"])
        self.assertIn("不构成临床", payload["analysis"])
        self.assertIn("ranking_csv", payload["files"])
        self.assertIn("analysis_report", payload["files"])


if __name__ == "__main__":
    unittest.main()
