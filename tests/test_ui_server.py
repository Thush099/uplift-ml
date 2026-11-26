"""Tests for the causal uplift lab UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_uplift


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/run",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_run_endpoint_trains_real_models(server):
    status, payload = _post(server, {"n_samples": 300})
    assert status == 200
    assert set(payload["models"].keys()) == {"S-learner", "T-learner", "X-learner"}
    for entry in payload["models"].values():
        report = entry["report"]
        assert "qini_coefficient" in report
        assert len(report["uplift_by_decile"]) == 10


def test_model_selection_is_respected():
    result = run_uplift(model_types=["X-learner"], n_samples=200)
    assert list(result["models"].keys()) == ["X-learner"]


def test_csv_upload_skips_synthetic_generation():
    csv_text = (
        "x1,treatment,outcome\n"
        + "\n".join(f"{i * 0.01},{i % 2},{(i * 7) % 3}" for i in range(120))
    )
    result = run_uplift(model_types=["T-learner"], csv_text=csv_text)
    assert list(result["models"].keys()) == ["T-learner"]
    assert result["n_rows"] == 120


def test_health_and_index_served(server):
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert health["model_types"] == ["S-learner", "T-learner", "X-learner"]
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "uplift" in html and "/api/run" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"model_types": ["Magic"]}, "model_type must be one of"),
        ({"n_samples": "x"}, "n_samples must be an integer"),
        ({"n_samples": 10}, "n_samples must be >= 50"),
        ({"csv_text": "a,b\n1,2", "outcome_col": "outcome"}, "column 'outcome' not found"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"


def test_index_has_loading_and_theme_states(server):
    """The page shows busy feedback during a run and follows the OS light/dark theme."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'aria-busy' in html, "form must announce busy state during a run"
    assert 'class="spinner"' in html, "missing visible busy indicator"
    assert "prefers-color-scheme" in html, "page must follow the OS light/dark theme"
