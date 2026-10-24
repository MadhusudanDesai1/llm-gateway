"""Request log. LOG_BACKEND=sqlite (local) or dynamodb (AWS)."""
from __future__ import annotations

import json
import os
import sqlite3
from decimal import Decimal
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


class SqliteStore:
    def __init__(self, path=None):
        self.path = path or os.getenv("SQLITE_PATH", "data/gateway.db")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, timestamp TEXT, prompt_hash TEXT, route TEXT,
                models_used TEXT, input_tokens INTEGER, output_tokens INTEGER,
                cost_usd REAL, strong_equiv_cost_usd REAL, latency_ms REAL,
                verifier_score REAL, verifier_mode TEXT, verifier_reason TEXT)"""
        )
        conn.commit()
        conn.close()

    def log(self, row: dict):
        row = {**row, "models_used": json.dumps(row["models_used"])}
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                """INSERT INTO requests VALUES (:request_id, :timestamp, :prompt_hash, :route,
                :models_used, :input_tokens, :output_tokens, :cost_usd, :strong_equiv_cost_usd,
                :latency_ms, :verifier_score, :verifier_mode, :verifier_reason)""",
                row,
            )
            conn.commit()
        finally:
            conn.close()

    def all(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY timestamp")]
        finally:
            conn.close()
        for r in rows:
            r["models_used"] = json.loads(r["models_used"])
        return rows


class DynamoStore:
    def __init__(self):
        import boto3

        self.table = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1")).Table(
            os.getenv("DDB_TABLE", "llm-gateway-requests")
        )

    def log(self, row: dict):
        self.table.put_item(Item=json.loads(json.dumps(row), parse_float=Decimal))

    def all(self):
        resp = self.table.scan()
        items = resp["Items"]
        while "LastEvaluatedKey" in resp:
            resp = self.table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items += resp["Items"]
        return [{k: (float(v) if isinstance(v, Decimal) else v) for k, v in i.items()} for i in items]


def get_store():
    return DynamoStore() if os.getenv("LOG_BACKEND", "sqlite") == "dynamodb" else SqliteStore()


def summarize(rows):
    if not rows:
        return {"requests": 0}
    n = len(rows)
    total = sum(r["cost_usd"] for r in rows)
    equiv = sum(r["strong_equiv_cost_usd"] for r in rows)
    out = {
        "requests": n,
        "escalation_rate": sum(r["route"] == "escalated" for r in rows) / n,
        "total_cost_usd": total,
        "strong_equiv_cost_usd": equiv,
        "est_savings_pct": (1 - total / equiv) * 100 if equiv else 0.0,
    }
    for route in ("cheap_only", "escalated"):
        lat = [r["latency_ms"] for r in rows if r["route"] == route]
        if lat:
            out[f"{route}_p50_ms"] = pct(lat, 50)
            out[f"{route}_p95_ms"] = pct(lat, 95)
    return out
