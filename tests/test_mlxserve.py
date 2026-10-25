"""
Self-test for the optional mlx-serve engine. No pytest, no weights, no real server:
    python tests/test_mlxserve.py
A stdlib fake mlx-serve answers /v1/load-model, /v1/unload-model and a streamed
/v1/images/generations, and records what it was sent. Covers the request mapping, the event
stream, errors, cancel, the loader's refusals, and the sampler end to end (hooks, load,
generate, unload, the mask composite).
"""
import base64
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from PIL import Image

import mlxserve_client as ms
import nodes

_checks = 0


def ok(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        raise AssertionError("FAIL: " + msg)


def png(w=64, h=48, color=(200, 40, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


class Fake:
    """A fake mlx-serve. ``events`` lists every call; ``mode`` picks the image endpoint's behaviour."""

    def __init__(self):
        self.events, self.bodies, self.mode = [], [], "ok"
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                fake.events.append(self.path)
                fake.bodies.append(body)
                if self.path in ("/v1/load-model", "/v1/unload-model"):
                    return self._json(200, {"model": {"id": "x", "state": "ready"}})
                if fake.mode == "404":
                    return self._json(404, {"error": {"message": "No model is registered at that path"}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                total = int(body.get("steps") or 4)
                for step in range(1, total + 1):
                    self._event({"type": "progress", "stage": "Generating", "step": step, "total": total})
                if fake.mode == "error":
                    return self._event({"type": "error", "error": "out of memory"})
                w, h = (int(v) for v in body["size"].split("x"))
                self._event({"type": "complete", "data": [{"b64_json": base64.b64encode(png(w, h)).decode()}]})

            def _event(self, ev):
                try:
                    self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


fake = Fake()

print("1. build_request maps the sampler's values")
b = ms.build_request("/p/klein", "a fox", seed=7, steps=0, guidance=0.0, width=1024, height=768)
ok(b["size"] == "1024x768" and b["seed"] == 7 and b["stream"] is True, "size, seed, stream")
ok("steps" not in b and "guidance_scale" not in b and "negative_prompt" not in b, "0 / empty leave the model default")
b = ms.build_request("/p/moss", "x", steps=25, guidance=2.5, negative_prompt="blurry")
ok(b["steps"] == 25 and b["guidance_scale"] == 2.5 and b["negative_prompt"] == "blurry", "set values are sent")
b = ms.build_request("/p/q", "edit", images=[b"one", b"two", b"three"])
ok(b["mode"] == "edit" and base64.b64decode(b["image"]) == b"one", "the first image is the one edited")
ok([base64.b64decode(r) for r in b["ref_images"]] == [b"two", b"three"], "the rest are extra references, once each")
ok("ref_images" not in ms.build_request("/p/q", "edit", images=[b"one"]), "a single image sends no ref_images")

print("2. generate follows the stream")
seen = []
out = ms.generate(fake.url, ms.build_request("/p/k", "x", steps=3, width=80, height=40), on_progress=lambda s, t: seen.append((s, t)))
ok(Image.open(io.BytesIO(out)).size == (80, 40), "the complete event's PNG is returned")
ok(seen == [(1, 3), (2, 3), (3, 3)], f"progress events reach the callback ({seen})")
fake.mode = "error"
try:
    ms.generate(fake.url, ms.build_request("/p/k", "x"))
    ok(False, "an error event raises")
except ms.MlxServeError as e:
    ok("out of memory" in str(e), "an error event is surfaced")
fake.mode = "404"
try:
    ms.generate(fake.url, ms.build_request("/p/k", "x"))
    ok(False, "HTTP 404 raises")
except ms.MlxServeError as e:
    ok("404" in str(e) and "registered" in str(e), "an HTTP error carries the server's message")
fake.mode = "ok"
try:
    ms.load("http://127.0.0.1:9", "/p/k", timeout=2)
    ok(False, "an unreachable server raises")
except ms.MlxServeError as e:
    ok("not reachable" in str(e) and "MLX_SERVE_URL" in str(e), "the unreachable message names MLX_SERVE_URL")
try:
    ms.generate(fake.url, ms.build_request("/p/k", "x", steps=6), interrupted=lambda: True)
    ok(False, "an interrupt raises")
except InterruptedError:
    ok(True, "an interrupt closes the stream")
os.environ.pop("MLX_SERVE_URL", None)
ok(ms.server_url() == ms.DEFAULT_URL, "default URL is mlx-serve's own")

print("3. the loader's mlx-serve path")
pack = tempfile.mkdtemp(prefix="pack-")
loader = nodes.MfluxModelLoader()
handle, = loader.load(nodes.PACK_DECIDES, "none", model_path=pack, keep_loaded=False, engine="mlx-serve")
ok(handle.engine == "mlx-serve" and handle.instance is None and handle.profile is None, "no mflux model or profile")
ok(handle.remote_model == os.path.realpath(pack), "the pack is named by its realpath")
ok(loader.VALIDATE_INPUTS(nodes.PACK_DECIDES) is True, "the pack-decides entry validates")
for kwargs, needle in ((dict(lora=[("/l.safetensors", 1.0)]), "LoRA"), (dict(controlnet_path="/c"), "ControlNet"),
                       (dict(quantize="4"), "quantize"), (dict(model_path="/no/such/dir"), "local pack folder")):
    args = dict(model=nodes.PACK_DECIDES, quantize="none", model_path=pack, engine="mlx-serve")
    args.update(kwargs)
    try:
        loader.load(**args)
        ok(False, f"{needle} is refused")
    except ValueError as e:
        ok(needle in str(e), f"{needle} is refused with a clear message")
for kwargs in (dict(model=nodes.PACK_DECIDES), dict(model="schnell", local_model=nodes.MLX_SERVE_TAG + "moss")):
    try:
        loader.load(quantize="none", engine="mflux", **kwargs)
        ok(False, "mlx-serve-only choices are refused on engine mflux")
    except ValueError as e:
        ok("engine: mlx-serve" in str(e), "mlx-serve-only choices name the engine to use")
ok("engine" in nodes.MfluxModelLoader.INPUT_TYPES()["optional"], "the engine widget exists")
ok(list(nodes.MfluxModelLoader.INPUT_TYPES()["optional"])[-1] == "engine", "engine is the last widget")

print("4. the sampler end to end against the fake server")
log = os.path.join(tempfile.mkdtemp(), "hooks.log")
os.environ["MLX_SERVE_URL"] = fake.url
os.environ["MFLUX_BEFORE_LOAD"] = f"echo before >> {log}"
os.environ["MFLUX_AFTER_FREE"] = f"echo after >> {log}"
fake.events.clear(); fake.bodies.clear()
image = torch.from_numpy(np.full((1, 40, 80, 3), 0.5, dtype=np.float32))
mask = torch.zeros((1, 40, 80, 3))
mask[:, :, :40, :] = 1.0   # the left half is locked to the original
payload = nodes.MfluxImagePayload(images=[image, image], mask=mask, strength=1.0)
out, info = nodes.MfluxModelSampler().generate(handle, "make it red", "auto", seed=3, steps=2, width=80, height=40,
                                               mflux_image=payload, mask_mode="preserve", mask_feather=0)
ok(fake.events == ["/v1/load-model", "/v1/images/generations", "/v1/unload-model"], f"load, generate, unload ({fake.events})")
ok(open(log).read().split() == ["before", "after"], "the memory hooks wrap the job")
sent = fake.bodies[1]
ok(sent["model"] == handle.remote_model and sent["mode"] == "edit" and len(sent["ref_images"]) == 1, "an edit with one extra reference")
ok("mask" not in sent, "the mask is never sent to the server")
ok(tuple(out.shape) == (1, 40, 80, 3), "IMAGE comes back at the requested size")
ok(abs(float(out[0, 20, 10, 0]) - 0.5) < 0.02 and float(out[0, 20, 70, 0]) > 0.7, "the mask composite keeps the locked half")
ok("forwarded=" in info and "'steps'" in info and "mask is applied as a composite" in info, "info names the request and the composite")
fake.events.clear(); fake.bodies.clear()
open(log, "w").close()
kept, = loader.load(nodes.PACK_DECIDES, "none", model_path=pack, keep_loaded=True, engine="mlx-serve")
nodes.MfluxModelSampler().generate(kept, "x", "auto", steps=1, width=64, height=48)
ok(fake.events == ["/v1/load-model", "/v1/images/generations"], "keep_loaded leaves the model in the server")
ok(open(log).read().split() == ["before"], "with keep_loaded the model stays in the server, so no after-hook (memory not given back)")
fake.mode = "error"
fake.events.clear(); fake.bodies.clear()
try:
    nodes.MfluxModelSampler().generate(handle, "x", "auto", steps=1, width=64, height=48)
    ok(False, "a failed job raises")
except ms.MlxServeError:
    ok(fake.events[-1] == "/v1/unload-model", "a failed job still unloads")

fake.mode = "ok"
_, info = nodes.MfluxModelSampler().generate(
    handle, "x", "auto", steps=1, width=64, height=48,
    mflux_image=nodes.MfluxImagePayload(images=[image], strength=0.6, strengths=[0.6]))
ok("image_strength" in info and "full strength" in info, "a strength below 1 is named as ignored, not dropped silently")

print(f"all {_checks} mlx-serve checks passed")
