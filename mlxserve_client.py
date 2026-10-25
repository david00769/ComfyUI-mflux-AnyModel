"""Run one image job on a local mlx-serve server (https://github.com/ddalcu/mlx-serve): load, generate, unload.

Stdlib only. The server is a separate install; its URL comes from ``MLX_SERVE_URL`` (default: mlx-serve's own
``http://127.0.0.1:11234``). A model is named by the absolute path of its pack folder, the same string for
``/v1/load-model``, the request and ``/v1/unload-model``.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Callable, Sequence

DEFAULT_URL = "http://127.0.0.1:11234"


class MlxServeError(RuntimeError):
    pass


def server_url() -> str:
    return (os.environ.get("MLX_SERVE_URL") or DEFAULT_URL).rstrip("/")


def _open(url: str, path: str, body: dict, timeout: float):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read()[:500].decode(errors="ignore")
        raise MlxServeError(f"mlx-serve {path} returned {e.code}: {detail}") from None
    except (urllib.error.URLError, OSError) as e:
        raise MlxServeError(f"mlx-serve is not reachable at {url} ({e}); start it, or set MLX_SERVE_URL") from None


def load(url: str, model: str, timeout: float = 600) -> dict:
    with _open(url, "/v1/load-model", {"model": model}, timeout) as r:
        return json.loads(r.read())


def unload(url: str, model: str, timeout: float = 60) -> None:
    with _open(url, "/v1/unload-model", {"model": model}, timeout) as r:
        r.read()


def build_request(model: str, prompt: str, *, negative_prompt: str = "", seed: int = 0, steps: int = 0,
                  guidance: float = 0.0, width: int = 1024, height: int = 1024,
                  images: Sequence[bytes] = ()) -> dict:
    """The /v1/images/generations body. 0 steps or guidance leaves the model's default; ``images`` (PNG bytes)
    make it an edit: the first is the image to edit, the rest are extra references."""
    body = {"model": model, "prompt": prompt, "size": f"{int(width)}x{int(height)}", "seed": int(seed),
            "stream": True, "response_format": "b64_json"}
    if steps and int(steps) > 0:
        body["steps"] = int(steps)
    if guidance and float(guidance) > 0:
        body["guidance_scale"] = float(guidance)
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    if images:
        body["mode"] = "edit"
        body["image"] = base64.b64encode(images[0]).decode()
        if len(images) > 1:
            body["ref_images"] = [base64.b64encode(i).decode() for i in images[1:]]
    return body


def generate(url: str, body: dict, *, on_progress: Callable[[int, int], None] | None = None,
             interrupted: Callable[[], bool] | None = None, timeout: float = 3600) -> bytes:
    """POST the body and follow its event stream; returns the PNG. Closing the stream cancels the job."""
    with _open(url, "/v1/images/generations", body, timeout) as stream:
        try:
            for raw in stream:
                if interrupted is not None and interrupted():
                    raise InterruptedError("cancelled")
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                kind = event.get("type")
                if kind == "progress" and on_progress is not None:
                    on_progress(int(event.get("step") or 0), int(event.get("total") or 0))
                elif kind == "complete":
                    try:
                        return base64.b64decode(event["data"][0]["b64_json"])
                    except (KeyError, IndexError, TypeError, ValueError):
                        raise MlxServeError(f"mlx-serve sent a complete event without an image: {str(event)[:200]}") from None
                elif kind == "error" or "error" in event:
                    raise MlxServeError(f"mlx-serve: {event.get('error') or event}")
        except InterruptedError:
            raise
        except OSError as e:  # a timeout or reset while the image was being made
            raise MlxServeError(f"mlx-serve stream from {url} broke off: {e}") from None
    raise MlxServeError("mlx-serve closed the stream without an image")
