import sys, json
from pathlib import Path
from types import SimpleNamespace
import pytest, torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import provenance as prov
from flops import PrefillFlopEstimate
import bench_efficiency as eff

@pytest.fixture(autouse=True)
def clean_state(tmp_path,monkeypatch):
    monkeypatch.setenv("SPINVL_FINGERPRINT_CACHE",str(tmp_path/'cache'))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    prov._RUN_INPUT_SNAPSHOT=None

def test_merger_actual_unequal_linear_dimensions():
    module=nn.Sequential(nn.LayerNorm(12),nn.Linear(12,8),nn.GELU(),nn.Linear(8,8))
    with PrefillFlopEstimate({"merger":module}) as fp:module(torch.randn(2,5,12))
    assert fp.result()["flops_components"]["linear_matmul"]==2*10*(12*8+8*8)

def test_mha_functional_projections_counted():
    mha=nn.MultiheadAttention(16,4,kdim=12,vdim=8,batch_first=True)
    with PrefillFlopEstimate({"mha":mha}) as fp:
        mha(torch.randn(1,3,16),torch.randn(1,5,12),torch.randn(1,5,8),need_weights=False)
    d=fp.result()["flops_components"]
    assert d["mha_projections"]==2*(3*16*16+5*12*16+5*8*16+3*16*16)
    assert d["attention_products"]==4*3*5*16

def test_missing_checkpoint_fails_before_loading():
    args=SimpleNamespace(modes="ours",joint_ckpt="missing.pt",tower_ckpt="",rose_ckpt="",proj_ckpt="",front_ckpt="",synthetic=False,gen_tokens=24,warm=0,runs=1,Ns="64")
    with pytest.raises(FileNotFoundError):eff.validate_args(args)
    args.joint_ckpt=""
    with pytest.raises(ValueError):eff.validate_args(args)

def test_input_bytes_and_all_checkpoint_roles(tmp_path):
    files=[]
    for role in ["joint_ckpt","tower_ckpt","rose_ckpt","proj_ckpt"]:
        p=tmp_path/(role+'.pt');p.write_bytes(role.encode());files.append((role,str(p)))
    args=dict(files)
    meta=tmp_path/'x.meta.pt';feat=tmp_path/'features.bin';feat.write_bytes(b'feat')
    torch.save({"feat_bin":str(feat)},meta)
    image=tmp_path/'image.png';image.write_bytes(b'rawimage')
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps({"id":"a","image":image.name}))
    args['source']=[f"{meta}|{manifest}|{tmp_path}"]
    snap=prov.capture_run_inputs(args)
    assert set(snap['ckpt_sha256'])==set(dict(files))
    assert snap['data_fingerprint']['file_count']==4
    image.write_bytes(b'newimage')
    with pytest.raises(RuntimeError,match='changed'):prov.build_run_manifest('test',args)

def test_backbone_bytes_and_optimizer_whitelist():
    model=nn.Linear(2,2).to(torch.bfloat16).requires_grad_(False)
    before,_=prov.snapshot_backbone_hashes(model)
    visual=nn.Parameter(torch.ones(2));other=nn.Parameter(torch.ones(2))
    opt=torch.optim.SGD([visual],lr=.1)
    prov.verify_frozen_backbone_and_optimizer(model,opt,[visual],before)
    with pytest.raises(AssertionError):prov.verify_frozen_backbone_and_optimizer(model,opt,[visual,other])
    with torch.no_grad():model.weight.view(torch.uint8).reshape(-1)[0]^=1
    with pytest.raises(AssertionError):prov.verify_frozen_backbone_and_optimizer(model,opt,[visual],before)

def test_flops_matches_tiny_qwen_and_vrmg():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from wrap import attach_rotation,rotation_context_with,freeze_all_but
    cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    model=Qwen3ForCausalLM(cfg).eval();mods,reg=attach_rotation(model,[0],block_size=8,visual_dim=12,query_dim=8,num_heads=2,multifreq=True,hybrid=True,qcond=True,direct_hyb=True)
    with torch.no_grad(),PrefillFlopEstimate({'model':model,'vrmg':mods[0]}) as fp:
        with rotation_context_with(reg,torch.randn(1,4,12),torch.ones(1,4)):
            model(input_ids=torch.tensor([[1,2,3]]))
    result=fp.result()
    assert result['flops_components']['cayley_lu_solve_estimate'] == pytest.approx(mods[0].total_blocks*(8/3)*8**3)
    assert result['flops_components']['rotation_products']>0
    assert mods[0].total_blocks==sum(mods[0].proj_dims.values())//8
