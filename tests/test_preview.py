"""Live-preview bridge: family->LatentCreator resolution, registry hygiene, decode contract.

No weights load here. The registry guard and the decode-tuple shape are the parts that, if wrong,
either leak callbacks across runs (progress bar counts N*runs) or crash a real generation, so they
are exercised against fakes that mimic mflux's registry and VAE.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import preview as P  # noqa: E402


def check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok  {name}")


print("resolve_latent_creator")
check("flux resolves", P.resolve_latent_creator("flux").__name__ == "FluxLatentCreator")
check("flux variant prefix resolves", P.resolve_latent_creator("flux-depth").__name__ == "FluxLatentCreator")
check("krea2 resolves", P.resolve_latent_creator("krea2").__name__ == "Krea2LatentCreator")
check("krea2-depth maps to krea2", P.resolve_latent_creator("krea2-depth").__name__ == "Krea2LatentCreator")
check("ideogram4 resolves", P.resolve_latent_creator("ideogram4").__name__ == "Ideogram4LatentCreator")
check("qwen resolves", P.resolve_latent_creator("qwen").__name__ == "QwenLatentCreator")
check("z-image resolves", P.resolve_latent_creator("z-image").__name__ == "ZImageLatentCreator")
check("flux2 resolves", P.resolve_latent_creator("flux2").__name__ == "Flux2LatentCreator")
check("unknown family degrades to None", P.resolve_latent_creator("boogu") is None)  # boogu has no creator
check("empty family is None", P.resolve_latent_creator("") is None)
check("None family is None", P.resolve_latent_creator(None) is None)


class FakeRegistry:
    def __init__(self):
        self.before_loop, self.in_loop, self.after_loop, self.interrupt = [], [], [], []

    def register(self, cb):
        for attr in ("before_loop", "in_loop", "after_loop", "interrupt"):
            if hasattr(cb, "call_" + attr):
                getattr(self, attr).append(cb)


class FakeInstance:
    def __init__(self):
        self.callbacks = FakeRegistry()


print("_RegistryGuard hygiene")
inst = FakeInstance()
existing = object()
inst.callbacks.in_loop.append(existing)  # pretend something was already registered
cb = P.ComfyLivePreview(inst, None)
with P._RegistryGuard(inst, cb):
    check("callback is registered inside the block", cb in inst.callbacks.in_loop)
    check("existing callbacks are preserved", existing in inst.callbacks.in_loop)
check("callback is removed after the block", cb not in inst.callbacks.in_loop)
check("registry restored to exactly its prior state", inst.callbacks.in_loop == [existing])

# repeated runs must not accumulate callbacks (the keep_loaded leak)
for _ in range(5):
    with P._RegistryGuard(inst, P.ComfyLivePreview(inst, None)):
        pass
check("no callback leak across 5 runs", inst.callbacks.in_loop == [existing])

print("guard survives a missing registry")
bare = type("Bare", (), {})()  # no .callbacks
with P._RegistryGuard(bare, P.ComfyLivePreview(bare, None)):
    pass
check("no crash when instance has no registry", True)

print("decode never breaks generation")
cb = P.ComfyLivePreview(type("M", (), {"vae": None})(), latent_creator=None)
check("decode with no latent_creator is guarded upstream (returns None on error)",
      cb._decode(object(), type("C", (), {"height": 512, "width": 512})()) is None)

print("decode path selection (the Ideogram 4 trap)")


class FakeVAE:
    """Mimics Flux2VAE: HAS decode_packed_latents, declares 32 latent channels."""
    latent_channels = 32

    def __init__(self):
        self.calls = []

    def decode_packed_latents(self, x):
        self.calls.append("packed"); return _img()

    def decode(self, x):
        self.calls.append("plain"); return _img()


class FakeCreator:
    out = None

    @classmethod
    def unpack_latents(cls, latents, height, width):
        return cls.out


def _img():
    """Image-shaped decoder output, so _decode completes through the PIL conversion."""
    import mlx.core as mx
    return mx.zeros((1, 3, 8, 8))


def Arr(ch):
    """A real array, so _decode runs to completion instead of dying after the branch."""
    import mlx.core as mx
    return mx.zeros((1, ch, 8, 8))


def path_for(channels):
    vae = FakeVAE()
    FakeCreator.out = Arr(channels)
    cb = P.ComfyLivePreview(type("M", (), {"vae": vae})(), FakeCreator)
    cb._decode(object(), type("C", (), {"height": 512, "width": 512})())
    return vae.calls[0] if vae.calls else "none"


# FLUX.2: unpack_latents leaves 128 patchified channels -> needs decode_packed_latents
check("128ch (FLUX.2, still patchified) -> decode_packed_latents", path_for(128) == "packed")
# Ideogram 4: unpack_latents already applied shift/scale and unpatchified -> plain decode
check("32ch (Ideogram 4, VAE-ready) -> plain decode", path_for(32) == "plain")
check("below latent_channels also uses plain decode", path_for(16) == "plain")


class NoPackedVAE:
    latent_channels = 16

    def __init__(self):
        self.calls = []

    def decode(self, x):
        self.calls.append("plain"); return _img()


def path_no_packed(channels):
    vae = NoPackedVAE()
    FakeCreator.out = Arr(channels)
    cb = P.ComfyLivePreview(type("M", (), {"vae": vae})(), FakeCreator)
    cb._decode(object(), type("C", (), {"height": 512, "width": 512})())
    return vae.calls[0] if vae.calls else "none"


check("a VAE without decode_packed_latents is unaffected", path_no_packed(128) == "plain")

# The callback must take every argument mflux's GenerationContext passes it. 0.22 added
# control_images to before_loop, and a strict signature failed every generation with a preview.
import inspect  # noqa: E402

try:
    from mflux.callbacks.generation_context import GenerationContext
    src = inspect.getsource(GenerationContext)
    for hook in ("call_before_loop", "call_in_loop"):
        passed = set()
        for block in src.split(f"subscriber.{hook}(")[1:]:
            passed |= {ln.split("=")[0].strip() for ln in block.split(")")[0].splitlines() if "=" in ln}
        params = inspect.signature(getattr(P.ComfyLivePreview, hook)).parameters
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        check(f"{hook} accepts what mflux passes ({sorted(passed)})", takes_any or passed <= set(params))
except ImportError:
    print("callback signature check skipped: mflux without GenerationContext")

print("callback contract with mflux's GenerationContext")
# call_before_loop/call_in_loop import comfy.utils for the progress bar; stub it so this runs without ComfyUI.
import types  # noqa: E402

_pbar_updates = []
comfy_stub = types.ModuleType("comfy")
comfy_stub.utils = types.SimpleNamespace(ProgressBar=lambda total, node_id=None: types.SimpleNamespace(
    update_absolute=lambda step, total, preview: _pbar_updates.append((step, preview))))
sys.modules.setdefault("comfy", comfy_stub)
sys.modules.setdefault("comfy.utils", comfy_stub.utils)

import mlx.core as mx  # noqa: E402

cfg = types.SimpleNamespace(num_inference_steps=3, height=32, width=32,
                            scheduler=types.SimpleNamespace(sigmas=mx.array([1.0, 0.75, 0.5, 0.0])))
cb = P.ComfyLivePreview(FakeInstance(), None)
cb.call_before_loop(seed=1, prompt="p", latents=mx.ones((1, 4, 8)), config=cfg,
                    canny_image=None, depth_image=None, control_images=None)
check("call_before_loop accepts control_images (passed by mflux >= 0.22)", cb.total == 3)
cb.call_in_loop(t=0, seed=1, prompt="p", latents=mx.zeros((1, 4, 8)), config=cfg, time_steps=None)
check("every step advances the progress bar", _pbar_updates[-1][0] == 1)


print("class-specific latent creators (Qwen-Image-2.1 shares the 'qwen' family labels with 1.x)")


def named(cls_name):
    """An instance whose class carries the given name, as resolve_latent_creator keys on it."""
    return type(cls_name, (), {})()


try:
    import mflux.models.qwen21  # noqa: F401
    HAS_QWEN21 = True
except ImportError:
    HAS_QWEN21 = False
if HAS_QWEN21:
    check("QwenImage21 uses Qwen21LatentCreator, not the 1.x creator",
          P.resolve_latent_creator("qwen", named("QwenImage21")).__name__ == "Qwen21LatentCreator")
    check("QwenImage21Edit uses QwenImage21LatentCreator",
          P.resolve_latent_creator("qwen-edit", named("QwenImage21Edit")).__name__ == "QwenImage21LatentCreator")
else:
    check("2.1 classes degrade to no preview on a runtime without them",
          P.resolve_latent_creator("qwen", named("QwenImage21")) is None)
check("a 1.x instance still gets the family's creator",
      P.resolve_latent_creator("qwen", named("QwenImage")).__name__ == "QwenLatentCreator")

print("x0 estimate for the live preview (Qwen-Image-2.1)")
cb = P.ComfyLivePreview(named("QwenImage21"), None)
check("QwenImage21 previews the x0 estimate", cb.estimate_x0)
check("other models preview raw latents", not P.ComfyLivePreview(named("Flux1"), None).estimate_x0)
x_start = mx.ones((1, 4, 8))
cb.call_before_loop(seed=1, prompt="p", latents=x_start, config=cfg,
                    canny_image=None, depth_image=None, control_images=None)
check("the starting latents are remembered", cb._prev_latents is x_start)
x_next = mx.zeros((1, 4, 8))
cb.call_in_loop(t=0, seed=1, prompt="p", latents=x_next, config=cfg, time_steps=None)
check("every step remembers its latents, previewed or not", cb._prev_latents is x_next)
# Flow matching: x_s = (1 - s) * x0 + s * noise, and one Euler step x_{t+1} = x_t + (s_{t+1} - s_t) * v
# with v = noise - x0. From the states on either side of the step, the estimate must recover x0 exactly.
x0 = mx.random.normal((1, 16, 64), key=mx.random.key(0))
noise = mx.random.normal((1, 16, 64), key=mx.random.key(1))
sig = cfg.scheduler.sigmas
x_t = (1 - sig[1]) * x0 + sig[1] * noise
x_t1 = x_t + (sig[2] - sig[1]) * (noise - x0)
cb._prev_latents = x_t
est = cb._x0_estimate(1, x_t1, cfg)
check("x0 estimate recovers the clean latents from one Euler step", mx.allclose(est, x0, atol=1e-4).item())
check("x0 estimate keeps the latents' dtype", est.dtype == x_t1.dtype)
cb._prev_latents = None
check("no previous state -> raw latents", cb._x0_estimate(1, x_t1, cfg) is x_t1)
check("no scheduler sigmas -> raw latents",
      cb._x0_estimate(1, x_t1, types.SimpleNamespace(num_inference_steps=3)) is x_t1)


print("\nPREVIEW OK — resolver + registry hygiene + decode safety verified")
