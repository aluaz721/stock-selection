import sqlite3
from pathlib import Path

import pandas as pd

from app.data.build_fundamentals_db import build_fundamentals_db


def make_csv(tmp_path, rows: dict) -> Path:
    csv_path = tmp_path / "fundamentals.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


class TestBuildFundamentalsDb:
    def test_creates_table_matching_csv(self, tmp_path):
        csv_path = make_csv(
            tmp_path, {"ticker": ["AAA", "BBB"], "datadate": ["2020-01-01", "2020-01-01"], "y_return": [0.01, -0.02]}
        )
        db_path = tmp_path / "test.db"

        result_path = build_fundamentals_db(csv_path=csv_path, db_path=db_path)

        assert result_path == db_path
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM fundamental_data", conn)
        conn.close()
        assert len(df) == 2
        assert set(df["ticker"]) == {"AAA", "BBB"}
        assert list(df.columns) == ["ticker", "datadate", "y_return"]

    def test_rerun_replaces_rather_than_appends(self, tmp_path):
        csv_path = make_csv(tmp_path, {"ticker": ["AAA"], "datadate": ["2020-01-01"]})
        db_path = tmp_path / "test.db"

        build_fundamentals_db(csv_path=csv_path, db_path=db_path)
        build_fundamentals_db(csv_path=csv_path, db_path=db_path)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM fundamental_data").fetchone()[0]
        conn.close()
        assert count == 1

    def test_creates_parent_directory_if_missing(self, tmp_path):
        csv_path = make_csv(tmp_path, {"ticker": ["AAA"], "datadate": ["2020-01-01"]})
        db_path = tmp_path / "nested" / "dir" / "test.db"

        build_fundamentals_db(csv_path=csv_path, db_path=db_path)

        assert db_path.exists()
