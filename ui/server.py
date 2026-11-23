"""Lightweight local web UI server for the causal uplift modeling lab.

Serves the uplift workbench in ``ui/index.html`` and exposes a JSON API that runs the
real ``UpliftPipeline`` benchmark. Standard library only. Start with
``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config  # noqa: E402
from src.pipeline import UpliftPipeline, run_benchmark  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

VALID_MODEL_TYPES = ("S-learner", "T-learner", "X-learner")
MAX_CSV_BYTES = 4_000_000


def run_uplift(
    model_types: Optional[Sequence[str]] = None,
    n_samples: int = 3000,
    seed: int = 42,
    csv_text: Optional[str] = None,
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
) -> dict[str, Any]:
    """Train and evaluate uplift models and return their reports.

    Without ``csv_text``, a synthetic heterogeneous dataset is generated so the lab works
    out of the box. Raises ValueError with a clear message for invalid inputs.
    """
    requested = tuple(model_types) if model_types else VALID_MODEL_TYPES
    for m in requested:
        if m not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {list(VALID_MODEL_TYPES)}, got {m!r}")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValueError(f"n_samples must be an integer, got {n_samples!r}")
    if n_samples < 50:
        raise ValueError(f"n_samples must be >= 50, got {n_samples}")

    if csv_text is None:
        df, results = run_benchmark(
            n_samples=n_samples,
            config=Config(random_state=seed),
            model_types=requested,
        )
    else:
        import pandas as pd

        if len(csv_text.encode("utf-8")) > MAX_CSV_BYTES:
            raise ValueError(f"csv_text exceeds {MAX_CSV_BYTES} bytes")
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as exc:
            raise ValueError(f"csv_text could not be parsed: {exc}") from exc
        for col in (outcome_col, treatment_col):
            if col not in df.columns:
                raise ValueError(f"column {col!r} not found in CSV")
        true_cate = df["true_cate"].to_numpy(dtype=float) if "true_cate" in df.columns else None
        pipeline = UpliftPipeline(config=Config(random_state=seed))
        results = {}
        for m in requested:
            results[m] = pipeline.run_on_dataframe(df, model_type=m, true_cate=true_cate)

    return {
        "n_rows": int(len(df)),
        "models": {
            name: {
                "model_type": res.model_type,
                "report": res.report.to_dict(),
                "feature_names": res.feature_names,
            }
            for name, res in results.items()
        },
    }


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok", "model_types": list(VALID_MODEL_TYPES)})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/run":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_uplift(
                    model_types=payload.get("model_types"),
                    n_samples=payload.get("n_samples", 3000),
                    seed=payload.get("seed", 42),
                    csv_text=payload.get("csv_text"),
                    outcome_col=payload.get("outcome_col", "outcome"),
                    treatment_col=payload.get("treatment_col", "treatment"),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the causal uplift lab UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8127)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Causal uplift lab running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())