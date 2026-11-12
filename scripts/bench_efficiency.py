"""Isolated inference timing with actual checkpoints and explicit FLOP estimates.

Each mode runs in a fresh subprocess/CUDA context. Use --synthetic explicitly
for random adapters; missing supplied checkpoints always fail. 'keep' is a
token-retention compute reference, not a trained VoRA reproduction.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from front import PatchFront
from flops import PrefillFlopEstimate
from provenance import build_run_manifest, capture_run_inputs
from wrap import attach_rotation, rotation_context_with

B = "/data/ibu-bi-ai-share-disk"
Q = 64
MODES = ("ours", "tower_ours", "rose", "keep", "llava")
CKPT_FOR = {"ours": "joint_ckpt", "tower_ours": "tower_ckpt", "rose": "rose_ckpt", "llava": "proj_ckpt"}


def grid_for(n):
    s = math.isqrt(n)
    if s*s != n or s%2:
        raise ValueError("Patch count must be an exact square with an even side length")
    return torch.tensor([[1,s,s]]), n


def validate_args(args):
    modes = args.modes.split(",")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError(f"Invalid modes: {args.modes}")
    for key in (*CKPT_FOR.values(), "front_ckpt"):
        value = getattr(args, key, "")
        if value and not Path(value).is_file():
            raise FileNotFoundError(f"--{key.replace('_','-')}: {value}")
    if args.front_ckpt:
        raise ValueError("Use a complete --joint-ckpt; partial front-only timing is unsupported")
    for mode in modes:
        if mode in CKPT_FOR and not args.synthetic and not getattr(args, CKPT_FOR[mode]):
            raise ValueError(f"{mode} requires --{CKPT_FOR[mode].replace('_','-')} (or explicit --synthetic)")
    if args.synthetic and any(getattr(args,k) for k in CKPT_FOR.values()):
        raise ValueError("Do not mix synthetic adapters with trained checkpoint arguments")
    if args.gen_tokens<1 or args.warm<0 or args.runs<1:
        raise ValueError("Invalid generation/timing counts")
    for n in args.Ns.split(","):
        grid_for(int(n))
    return modes


def run_prefill_and_decode(prefill_fn, step_fn, gen_tokens, warm=2, runs=8):
    def one_trial():
        with ExitStack() as stack:
            torch.cuda.synchronize(); t0=time.perf_counter()
            token, cache = prefill_fn(stack)
            torch.cuda.synchronize(); t1=time.perf_counter()
            for _ in range(gen_tokens-1):
                token, cache = step_fn(token,cache)
            torch.cuda.synchronize(); t2=time.perf_counter()
        return {"prefill_ms":(t1-t0)*1000, "decode_ms":(t2-t1)*1000, "total_ms":(t2-t0)*1000}
    for _ in range(warm):
        one_trial()
    torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated()/1024**3
    trials=[one_trial() for _ in range(runs)]
    result={"trials":trials, "resident_gib":baseline,
            "peak_gib":torch.cuda.max_memory_allocated()/1024**3,
            "timing_definition":"prefill includes token 1; decode generates tokens 2..N; paired per-trial times"}
    for key in ("prefill_ms","decode_ms","total_ms"):
        values=[t[key] for t in trials]
        result[key]=statistics.median(values)
        result[key.replace("_ms","_mean_ms")]=statistics.mean(values)
        result[key.replace("_ms","_std_ms")]=statistics.stdev(values) if runs>1 else 0.0
    result["ms"]=result["prefill_ms"]
    # Separate medians need not add; raw paired trials and means do.
    return result


def _build(args, mode):
    from bench_models import Runner
    if mode != "keep" and not args.synthetic:
        api_mode={"ours":"tower_free","tower_ours":"tower","rose":"tower","llava":"proj"}[mode]
        runner=Runner(api_mode,getattr(args,CKPT_FOR[mode]),args.model,teacher=args.teacher)
        if runner.arch.get("tuned_backbone"):
            raise ValueError("Efficiency comparison requires an unmodified frozen original LLM")
        if mode=="rose" and any(runner.arch.get(k) for k in ("multifreq","hybrid","qcond","direct_hyb","no_rot")):
            raise ValueError("ROSE checkpoint is not the single-frequency, rotation-only baseline")
        if mode in ("ours","tower_ours") and (not all(runner.arch.get(k) for k in ("multifreq","hybrid","qcond","direct_hyb")) or runner.arch.get("no_rot")):
            raise ValueError("Checkpoint does not implement the requested full SpinVL architecture")
        if mode=="ours" and (runner.arch["K"]!=args.K or runner.arch["front_mode"]!="rots"):
            raise ValueError("Joint checkpoint frontend differs from --K/rots")
        return runner.model,runner.registry,runner.front,runner.vision,runner.proj,runner.arch
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map="cuda").eval()
    registry={};front=tower=proj=None
    if mode in ("ours","tower_ours","rose"):
        layers=[0,4,8,12,16,20,24]
        full=mode!="rose"
        modules,registry=attach_rotation(model,layers,block_size=64,visual_dim=4096,query_dim=256,
            wrap_extra=[i for i in range(args.K) if i not in layers] if mode=="ours" else [],
            multifreq=full,hybrid=full,qcond=full,direct_hyb=full)
        for m in modules:m.eval()
        if mode=="ours":front=PatchFront(model,K=args.K,mode="rots",vdim=4096).cuda().eval()
    if mode in ("tower_ours","rose","llava"):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        cfg=AutoConfig.from_pretrained(args.teacher).vision_config
        tower=Qwen3VLVisionModel(cfg).cuda().to(torch.bfloat16).eval()
    if mode=="llava":
        from train_proj import Proj
        proj=Proj(4096,model.config.hidden_size).cuda().float().eval()
    return model,registry,front,tower,proj,{"synthetic_adapters":args.synthetic,"reference_only":mode=="keep"}


def run_worker(args,mode):
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for latency measurement")
    capture_run_inputs(vars(args))
    torch.manual_seed(args.seed)
    model,registry,front,tower,proj,arch=_build(args,mode)
    for p in model.parameters():p.requires_grad_(False)
    txt=torch.randint(100,1000,(1,Q),device="cuda"); mask=torch.ones_like(txt)
    roots={"llm":model,"front":front,"vision":tower,"projector":proj}
    roots.update({f"vrmg_{k}":v for k,v in registry.items()})
    rows=[]
    with torch.inference_mode():
        for count in args.Ns.split(","):
            grid,n=grid_for(int(count)); cells=n//4
            patches=torch.randn(1,n,1536,device="cuda",dtype=torch.float16)
            retained=torch.randn(1,cells,model.config.hidden_size,device="cuda",dtype=model.dtype) if mode=="keep" else None
            def prefill(stack):
                if mode=="ours":
                    features,lens=front(patches,grid,[n]); vf=features[:,:lens[0]].float()
                elif tower is not None:
                    out=tower(patches[0].to(torch.bfloat16),grid_thw=grid.to("cuda"))
                    vf=getattr(out,"pooler_output",None)
                    if vf is None:vf=out[0] if isinstance(out,(tuple,list)) else out
                    vf=vf.float().unsqueeze(0)
                if registry:
                    visual_mask=torch.ones(1,vf.shape[1],device="cuda")
                    stack.enter_context(rotation_context_with(registry,vf,visual_mask,text_mask=mask))
                    out=model(input_ids=txt,attention_mask=mask,use_cache=True)
                else:
                    v=retained if mode=="keep" else proj(vf[0]).to(model.dtype).unsqueeze(0)
                    embeddings=torch.cat([v,model.get_input_embeddings()(txt)],dim=1)
                    out=model(inputs_embeds=embeddings,use_cache=True)
                return out.logits[:,-1:,:].argmax(-1),out.past_key_values
            def step(token,cache):
                out=model(input_ids=token,past_key_values=cache,use_cache=True)
                return out.logits[:,-1:,:].argmax(-1),out.past_key_values
            # Shape profiling is untimed and uses the same prefill implementation.
            with PrefillFlopEstimate(roots) as estimator,ExitStack() as stack:
                token,cache=prefill(stack)
                del token,cache
            estimate=estimator.result()
            stats=run_prefill_and_decode(prefill,step,args.gen_tokens,args.warm,args.runs)
            rows.append({"N_patches":n,"cells":cells,"gen_tokens":args.gen_tokens,**estimate,**stats})
    result={"mode":mode,"pid":os.getpid(),"arch":arch,"rows":rows,
        "comparison_role":"token_retention_compute_reference" if mode=="keep" else "synthetic_architecture" if args.synthetic else "trained_checkpoint",
        "input_kind":"random_preprocessed_patches; fixed 64-token text prefix",
        "manifest":build_run_manifest("efficiency_worker",vars(args))}
    Path(args.worker_out).write_text(json.dumps(result,indent=2),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--teacher",default=f"{B}/zk/model/Qwen3-VL-8B-Instruct")
    ap.add_argument("--K",type=int,default=4);ap.add_argument("--Ns",default="256,576,1024,2304,4096")
    ap.add_argument("--gen-tokens",type=int,default=24);ap.add_argument("--modes",default=",".join(MODES))
    for key in (*CKPT_FOR.values(),"front_ckpt"):ap.add_argument("--"+key.replace("_","-"),default="")
    ap.add_argument("--synthetic",action="store_true");ap.add_argument("--warm",type=int,default=2)
    ap.add_argument("--runs",type=int,default=8);ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--worker-mode",choices=MODES,default="",help=argparse.SUPPRESS)
    ap.add_argument("--worker-out",default="",help=argparse.SUPPRESS)
    ap.add_argument("--out",required=True)
    args=ap.parse_args(); modes=validate_args(args)
    if args.worker_mode:
        run_worker(args,args.worker_mode);return
    results={}
    with tempfile.TemporaryDirectory(prefix="spinvl_eff_") as temp:
        for mode in modes:
            output=str(Path(temp)/(mode+".json"))
            subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),*sys.argv[1:],
                            "--worker-mode",mode,"--worker-out",output],check=True)
            results[mode]=json.loads(Path(output).read_text(encoding="utf-8"))
    rows=[]
    for i,count in enumerate(args.Ns.split(",")):
        n=int(count)
        rows.append({"N_patches":n,"cells":n//4,**{m:results[m]["rows"][i] for m in modes}})
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"schema_version":3,"Q":Q,"K":args.K,"rows":rows,
        "modes":results,"note":"Fresh subprocess per mode; algorithmic FLOP estimate, not measured hardware FLOPs. Paired trials retained; separate medians need not add."},indent=2),encoding="utf-8")
    print("BENCH_DONE",out)


if __name__=="__main__":main()
