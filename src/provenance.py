"""Content fingerprints and raw-byte frozen-backbone verification."""
import glob
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
TRACKED_REL_FILES = (
    "src/vrmg.py", "src/front.py", "src/wrap.py", "src/imgdata.py",
    "src/provenance.py", "src/flops.py", "scripts/bench_data.py",
    "scripts/bench_models.py", "scripts/bench_score.py", "scripts/eval_benchmarks.py",
    "scripts/train_final.py", "scripts/train_front.py", "scripts/train_joint.py",
    "scripts/train_opd.py", "scripts/train_joint_opd.py", "scripts/bench_efficiency.py",
)
CHECKPOINT_KEYS = ("vrmg_ckpt", "front_ckpt", "joint_ckpt", "init_ckpt", "ckpt",
                   "tower_ckpt", "rose_ckpt", "proj_ckpt")
PROMPT_PROTOCOL_VERSION = "clean_v6_unified_greedy_max24_no_ban_v2"
_RUN_INPUT_SNAPSHOT = None

def compute_code_hashes(root=ROOT):
    return {rel: hashlib.sha256((Path(root)/rel).read_bytes()).hexdigest() for rel in TRACKED_REL_FILES}

def _signature(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]

def _cache_file(path, kind="sha256"):
    directory = Path(os.environ.get("SPINVL_FINGERPRINT_CACHE", ROOT/"results/.fingerprint_cache"))
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    return directory/f"{key}.{kind}.json"

def _write_cache(path, obj):
    tmp = path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def fast_file_sha256(path):
    """Full byte hash, cached by path/size/mtime/ctime/inode; not a sampled digest."""
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise ValueError(f"Expected file: {p}")
    before, cache = _signature(p), _cache_file(p)
    try:
        old = json.loads(cache.read_text(encoding="utf-8"))
        if old.get("signature") == before and old.get("path") == str(p):
            return old["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    if _signature(p) != before:
        raise RuntimeError(f"Input changed while hashing: {p}")
    _write_cache(cache, {"path": str(p), "signature": before, "sha256": h.hexdigest()})
    return h.hexdigest()

def _files_fingerprint(paths, compact=False):
    paths = sorted({Path(p).resolve(strict=True) for p in paths})
    def entry(p):
        return {"path": str(p), "size": p.stat().st_size, "sha256": fast_file_sha256(p)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(entry, paths))
    raw = json.dumps(files, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    if not compact:
        return {"sha256": digest, "files": files}
    directory = _cache_file(ROOT / "data", "inventory").parent
    inventory = directory / ("data-"+digest+".json")
    if not inventory.exists():
        tmp = inventory.with_name(inventory.name+f".{os.getpid()}.tmp")
        tmp.write_bytes(raw); os.replace(tmp, inventory)
    if fast_file_sha256(inventory) != digest:
        raise RuntimeError(f"Corrupt input inventory: {inventory}")
    return {"sha256": digest, "file_count": len(files), "files_manifest": str(inventory),
            "files_manifest_sha256": digest}


def _expand(spec):
    out = []
    for pattern in str(spec).split(","):
        if not pattern.strip():
            continue
        matches = sorted(glob.glob(pattern.strip()))
        if not matches:
            raise FileNotFoundError(f"No inputs match: {pattern}")
        out.extend(Path(p) for p in matches)
    return out

def _feature_files(meta_path):
    digest, cache = fast_file_sha256(meta_path), _cache_file(meta_path, "features")
    try:
        old = json.loads(cache.read_text(encoding="utf-8"))
        if old.get("meta_sha256") == digest:
            return [Path(p) for p in old["paths"]]
    except (OSError, ValueError, KeyError):
        pass
    meta = torch.load(meta_path, map_location="cpu", weights_only=False, mmap=True)
    paths = []
    if isinstance(meta, dict) and meta.get("feat_bin"):
        p = Path(meta["feat_bin"])
        paths.append((p if p.is_absolute() else Path(meta_path).parent/p).resolve(strict=True))
    del meta
    _write_cache(cache, {"meta_sha256": digest, "paths": [str(p) for p in paths]})
    return paths

def fingerprint_inputs(args_dict):
    checkpoints = {k: fast_file_sha256(args_dict[k]) for k in CHECKPOINT_KEYS if args_dict.get(k)}
    metas, files = [], []
    for key in ("data", "test_data"):
        if args_dict.get(key):
            metas.extend(_expand(args_dict[key]))
    sources = args_dict.get("source") or []
    if isinstance(sources, str):
        sources = [sources]
    eval_sources = args_dict.get("eval_source") or []
    if isinstance(eval_sources, str):
        eval_sources = [eval_sources]
    sources = list(sources) + list(eval_sources)
    for source in sources:
        meta_spec, manifest, _ = source.split("|", 2)
        metas.extend(_expand(meta_spec)); files.append(Path(manifest))
        image_root = source.split("|", 2)[2]
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                image = json.loads(line).get("image")
                if image:
                    files.append(Path(image_root)/image)
    files.extend(metas)
    for spec in args_dict.get("benchmark_assets", []):
        files.extend(_expand(spec))
    for p in sorted(set(metas)):
        if p.name.endswith(".meta.pt"):
            files.extend(_feature_files(p))
    models = {}
    for key in ("model", "teacher"):
        if not args_dict.get(key):
            continue
        p = Path(args_dict[key]).resolve(strict=True)
        if p.is_file():
            assets = [p]
        else:
            weights = list(p.glob("*.safetensors"))+list(p.glob("pytorch_model*.bin"))
            if not weights:
                raise FileNotFoundError(f"No local model weights for {key}: {p}")
            assets = weights+[q for q in p.iterdir() if q.is_file() and q.suffix in (".json", ".txt", ".tiktoken", ".model")]
        models[key] = {"identifier": str(p), **_files_fingerprint(assets)}
    return {"ckpt_sha256": checkpoints, "data_fingerprint": _files_fingerprint(files, compact=True), "model_fingerprints": models}

def capture_run_inputs(args_dict):
    """Capture before loading/training; reject later changes when emitting the final manifest."""
    global _RUN_INPUT_SNAPSHOT
    _RUN_INPUT_SNAPSHOT = {"code_sha256": compute_code_hashes(), **fingerprint_inputs(args_dict)}
    return _RUN_INPUT_SNAPSHOT

def snapshot_backbone_hashes(model):
    per_tensor, agg = {}, hashlib.sha1()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0].replace(".rot.base.", ".")):
        name = name.replace(".rot.base.", ".")
        raw = p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        sh = hashlib.sha1(raw).hexdigest(); per_tensor[name] = sh
        agg.update(f"{name}:{p.dtype}:{tuple(p.shape)}:{sh};".encode())
    return per_tensor, agg.hexdigest()

def verify_frozen_backbone_and_optimizer(model, opt=None, allowed_params=None, pre_hashes=None):
    req = [n for n,p in model.named_parameters() if p.requires_grad]
    if req:
        raise AssertionError(f"Frozen backbone has trainable parameters: {req[:5]}")
    model_ids = {id(p) for p in model.parameters()}
    if opt is not None:
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        if len(set(ids)) != len(ids) or set(ids)&model_ids:
            raise AssertionError("Optimizer contains duplicate or original LLM parameters")
        if allowed_params is not None and set(ids) != {id(p) for p in allowed_params}:
            raise AssertionError("Optimizer differs from the exact visual-parameter whitelist")
    post_agg = ""
    if pre_hashes is not None:
        post, post_agg = snapshot_backbone_hashes(model)
        if set(post) != set(pre_hashes) or any(post[k] != v for k,v in pre_hashes.items()):
            raise AssertionError("Original LLM parameter bytes changed")
    return {"frozen_verified": True, "backbone_tensor_count": len(model_ids),
            "changed_backbone_tensors": 0, "backbone_sha1": post_agg,
            "backbone_hash_format": "sha1_raw_tensor_bytes_v1"}

def build_run_manifest(stage, args_dict, split_info=None, extra=None):
    import transformers
    import bench_score
    current = {"code_sha256": compute_code_hashes(), **fingerprint_inputs(args_dict)}
    if _RUN_INPUT_SNAPSHOT is not None and current != _RUN_INPUT_SNAPSHOT:
        changed = [k for k in current if current[k] != _RUN_INPUT_SNAPSHOT[k]]
        raise RuntimeError(f"Run inputs/code changed: {changed}")
    manifest = {"schema_version": 3, "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": args_dict, "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "scorer_version": bench_score.SCORER_VERSION, **current,
        "inputs_captured_before_run": _RUN_INPUT_SNAPSHOT is not None,
        "hash_cache_validation": "resolved_path,size,mtime_ns,ctime_ns,inode",
        "split_info": split_info or {},
        "env": {"python": platform.python_version(), "torch": str(torch.__version__),
                "transformers": transformers.__version__, "cuda": torch.version.cuda or "none",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}
    if extra:
        overlap = set(extra)&set(manifest)
        if overlap:
            raise ValueError(f"extra cannot override provenance fields: {sorted(overlap)}")
        manifest.update(extra)
    return manifest
