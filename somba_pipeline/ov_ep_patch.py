"""
ov_ep_patch.py — Force and configure OpenVINO EP for ONNX Runtime sessions.

Usage (import BEFORE any code that creates ORT sessions):
    import ov_ep_patch
    ov_ep_patch.enable_openvino_gpu({
        "device_type": "GPU",      # or "AUTO:GPU,CPU", "GPU.0"
        "precision":   "FP16",
        "num_streams": "4",
        "cache_dir":   "/tmp/ov_cache",
    })
    # or just rely on env vars, including OPENVINO_FORCE=1

Then import your inference library:
    from inference import InferencePipeline
"""

from __future__ import annotations
import os
import copy
import onnxruntime as ort

_PATCHED = False
_ORIG_IS = ort.InferenceSession

def _env_default(key: str, default: str | None) -> str | None:
    v = os.getenv(key, default)
    return v if (v is not None and v != "") else None

def _normalize_providers_and_opts(providers, provider_options):
    """
    Return (names: list[str], opts: list[dict]) in the same order, preserving any
    provider-specific options that may have been passed as tuples in 'providers'
    or as a parallel 'provider_options' list.
    """
    names = []
    opts = []

    # 1) Pull names & embedded options from 'providers'
    if providers:
        for p in list(providers):
            if isinstance(p, str):
                names.append(p)
                opts.append({})
            elif isinstance(p, (list, tuple)) and len(p) >= 1:
                name = p[0]
                embedded = (p[1] if len(p) > 1 and isinstance(p[1], dict) else {})
                names.append(name if isinstance(name, str) else str(name))
                opts.append(copy.deepcopy(embedded))
            else:
                names.append(str(p))
                opts.append({})
    else:
        names = []
        opts = []

    # 2) Merge/align with 'provider_options'
    # ORT requires provider_options as list[dict] matching providers by index.
    if provider_options is None:
        ext = [{} for _ in range(len(names))]
    elif isinstance(provider_options, dict):
        # Rare: dict keyed by provider names
        ext = [copy.deepcopy(provider_options.get(n, {})) for n in names]
    else:
        ext = list(provider_options)
        if len(ext) < len(names):
            ext += [{} for _ in range(len(names) - len(ext))]
        # coerce every slot to dict
        for i in range(len(ext)):
            if not isinstance(ext[i], dict):
                ext[i] = {}

    # 3) Combine embedded opts (from tuples) with external list opts (caller wins)
    for i in range(len(names)):
        merged = dict(opts[i])
        merged.update(ext[i])
        opts[i] = merged

    return names, opts

def enable_openvino_gpu(options: dict[str, str] | None = None,
                        disable_ort_graph_optim: bool = True,
                        default_providers: list[str] | None = None,
                        verbose: bool | None = None) -> None:
    """
    Install a monkey-patch that:
      - Forces-inserts OpenVINOExecutionProvider if requested (OPENVINO_FORCE=1),
      - Injects OpenVINO provider_options while keeping providers as strings,
      - Disables ORT graph optimizations when OV EP is present.

    Args:
      options: OV EP key/values (strings). Env defaults are used if not provided.
      disable_ort_graph_optim: When True, sets graph opts to ORT_DISABLE_ALL if OV present.
      default_providers: Used when caller provides none. Default: ["OpenVINOExecutionProvider","CPUExecutionProvider"].
      verbose: Print providers per session. Default from env OV_PATCH_VERBOSE=1.
    """
    global _PATCHED
    if _PATCHED:
        return

    if default_providers is None:
        default_providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    # Respect env-based provider discovery that some libs use
    os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", ",".join(default_providers))
    os.environ.setdefault("REQUIRED_ONNX_PROVIDERS", "OpenVINOExecutionProvider")

    # Build OV options: env as defaults, then override with 'options'
    ov_opts = {
        "device_type": _env_default("OPENVINO_DEVICE_TYPE", "GPU"),
        "precision":   _env_default("OPENVINO_PRECISION", "FP16"),
        "num_streams": _env_default("OPENVINO_NUM_STREAMS", "4"),
        "num_of_threads": _env_default("OPENVINO_NUM_THREADS", None),
        "cache_dir":   _env_default("OPENVINO_CACHE_DIR", "/tmp/ov_cache"),
        "enable_qdq_optimizer": _env_default("OPENVINO_ENABLE_QDQ_OPTIMIZER", None),
        "load_config": _env_default("OPENVINO_LOAD_CONFIG", None),
    }
    if options:
        for k, v in options.items():
            if v is not None and v != "":
                ov_opts[k] = str(v)
    # strip empties
    ov_opts = {k: v for k, v in ov_opts.items() if v is not None and v != ""}

    force_ov = os.getenv("OPENVINO_FORCE", "0").lower() in ("1", "true", "yes")
    _verbose = verbose if verbose is not None else (os.getenv("OV_PATCH_VERBOSE") == "1")

    OrigIS = _ORIG_IS

    def _patched_InferenceSession(*args, **kwargs):
        # Normalize providers and any caller-specified provider_options
        names, p_opts = _normalize_providers_and_opts(kwargs.get("providers"), kwargs.get("provider_options"))

        # If the caller passed no providers at all, use defaults
        if not names:
            names = list(default_providers)
            p_opts = [{} for _ in names]

        # Force-insert OpenVINO EP at the front if requested and missing
        if force_ov and "OpenVINOExecutionProvider" not in names:
            names.insert(0, "OpenVINOExecutionProvider")
            p_opts.insert(0, dict(ov_opts))

        # Inject/merge OV options for the OV slot (do not clobber user-provided keys)
        for i, n in enumerate(names):
            if n == "OpenVINOExecutionProvider":
                merged = dict(ov_opts)
                merged.update(p_opts[i])  # caller has priority
                p_opts[i] = merged

        kwargs["providers"] = names
        kwargs["provider_options"] = p_opts

        # Disable ORT graph optimizations when OV is present (Intel guidance)
        so = kwargs.get("sess_options") or ort.SessionOptions()
        if disable_ort_graph_optim and any(n == "OpenVINOExecutionProvider" for n in names):
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        kwargs["sess_options"] = so

        sess = OrigIS(*args, **kwargs)
        if _verbose:
            try:
                print("[ov_ep_patch] session providers:", sess.get_providers())
            except Exception:
                pass
        return sess

    ort.InferenceSession = _patched_InferenceSession
    _PATCHED = True
