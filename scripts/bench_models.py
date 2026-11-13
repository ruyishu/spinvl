"""评测用的四种模型形态：tower / tower_free / proj / base。
所有模型采用完全统一的贪心生成协议：取消按榜单首 token 禁词，统一提示、长度、EOS 与重复限制。
"""

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgdata  # noqa: E402
import train_opd as topd  # noqa: E402
from front import PatchFront  # noqa: E402
from wrap import attach_rotation, freeze_all_but, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"


def load_vision_tower(teacher, dev):
    from transformers import AutoModelForImageTextToText
    full = AutoModelForImageTextToText.from_pretrained(teacher, dtype=torch.bfloat16, trust_remote_code=True).eval()
    vis = getattr(full, "visual", None) or full.model.visual
    vis = vis.to(dev).eval()
    for p in vis.parameters():
        p.requires_grad_(False)
    del full
    gc.collect()
    return vis


class Runner:
    def __init__(self, mode, ckpt, model_path, teacher=imgdata.TEACHER, dev="cuda", flags=(),
                 prompt_alpha=None, hyb_bound=None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        self.mode, self.dev = mode, dev
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.stops = topd.stop_ids(self.tok)
        self.pad_id = self.tok.pad_token_id
        self.let_ids = {l: self.tok.encode(" " + l, add_special_tokens=False)[0] for l in "ABCDEFGH"}
        self.yn_ids = {"Yes": self.tok.encode(" Yes", add_special_tokens=False)[0],
                       "No": self.tok.encode(" No", add_special_tokens=False)[0]}
        self.step0_ban_ids = sorted({
            self.tok.encode(s, add_special_tokens=False)[0]
            for s in [" A", " B", " C", " D", " E", " F", " G", " H", "A", "B", "C", "D", "E", "F", "G", "H", " (A)", " (B)", " (C)", " (D)"]
            if len(self.tok.encode(s, add_special_tokens=False)) == 1
        })
        self.ip = AutoProcessor.from_pretrained(teacher, trust_remote_code=True).image_processor
        self.vdim = AutoConfig.from_pretrained(teacher, trust_remote_code=True).text_config.hidden_size
        self.registry, self.front, self.vision, self.proj = {}, None, None, None
        if mode == "teacher":
            from transformers import AutoModelForImageTextToText
            self.proc = AutoProcessor.from_pretrained(teacher, trust_remote_code=True)
            self.proc.tokenizer.padding_side = "left"
            self.model = AutoModelForImageTextToText.from_pretrained(teacher, dtype=torch.bfloat16, device_map=dev,
                                                                     trust_remote_code=True).eval()
            freeze_all_but(self.model, [])
            self.arch = {"teacher": True}
            return
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=dev,
                                                          trust_remote_code=True).eval()
        freeze_all_but(self.model, [])
        if mode == "base":
            self.arch = {"base": True, "model_path": model_path}
            return
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if mode in ("tower", "tower_free"):
            if mode == "tower":
                a1, ja, K, front_mode = ck["args"], {}, 0, "none"
                states = ck["state"]
            else:
                ja = ck["args"]
                a1 = torch.load(ja["vrmg_ckpt"], map_location="cpu", weights_only=False)["args"]
                front_mode = ja.get("front_mode", "rots")
                K = 0 if front_mode == "none" else int(ja.get("K", 4))
                states = ck["vrmg"]
            layers = [int(x) for x in a1["layers"].split(",") if x]
            kw = {f: bool(ja.get(f, False) or a1.get(f, False) or f"--{f}" in flags or f"--{f.replace('_', '-')}" in flags)
                  for f in ("multifreq", "tokrot", "hybrid", "qcond", "direct_hyb", "no_rot")}
            pa = float(prompt_alpha) if prompt_alpha is not None else float(ja.get("prompt_alpha", a1.get("prompt_alpha", 1.0)))
            hb = float(hyb_bound) if hyb_bound is not None else float(ja.get("hyb_bound", a1.get("hyb_bound", 0.0)))
            rb = float(ja.get("rot_bound", a1.get("rot_bound", 0.0)))
            vrmgs, self.registry = attach_rotation(self.model, layers, block_size=a1["block"], visual_dim=self.vdim,
                                                   query_dim=a1["query_dim"],
                                                   wrap_extra=[l for l in range(K) if l not in layers],
                                                   prompt_alpha=pa, hyb_bound=hb, rot_bound=rb, **kw)
            assert len(vrmgs) == len(states), f"vrmg count mismatch: {len(vrmgs)} != {len(states)}"
            for m, sd in zip(vrmgs, states):
                m.load_state_dict(sd)
            for m in vrmgs:
                m.eval()
            if ck.get("backbone"):
                self.model.load_state_dict(ck["backbone"], strict=True)
            self.arch = {"layers": layers, "block": a1["block"], **kw, "front_mode": front_mode, "K": K,
                         "prompt_alpha": pa, "hyb_bound": hb, "rot_bound": rb,
                         "tuned_backbone": bool(ck.get("backbone"))}
            if mode == "tower":
                self.vision = load_vision_tower(teacher, dev)
            else:
                self.front = PatchFront(self.model, K=K, mode=front_mode, vdim=self.vdim,
                                        lora_rank=ja.get("lora_rank", 16)).to(dev)
                self.front.load_state_dict(ck["front"])
                self.front.eval()
        elif mode in ("proj", "proj_pool32", "proj_fastv32"):
            from train_proj import Proj
            self.vision = load_vision_tower(teacher, dev)
            self.proj = Proj(self.vdim, self.model.config.hidden_size).to(dev).float()
            self.proj.load_state_dict(ck["proj"])
            self.proj.eval()
            self.arch = {"proj": mode}
        else:
            raise ValueError(mode)

    def _pix(self, images):
        pvs, grids, lens = [], [], []
        for img in images:
            o = self.ip(images=[img], return_tensors="pt")
            pvs.append(o["pixel_values"].to(torch.float16))
            grids.append(o["image_grid_thw"][0])
            lens.append(o["pixel_values"].shape[0])
        pv = torch.zeros(len(pvs), max(lens), pvs[0].shape[1], dtype=torch.float16)
        for j, p in enumerate(pvs):
            pv[j, : lens[j]] = p
        return pv, torch.stack(grids), lens

    @torch.no_grad()
    def _vision_feats(self, pv, grid, lens):
        rows = torch.cat([pv[k, : lens[k]] for k in range(len(lens))], dim=0)
        o = self.vision(rows.to(self.dev), grid_thw=grid.to(self.dev))
        vf = getattr(o, "pooler_output", None)
        if vf is None:
            vf = o[0] if isinstance(o, (tuple, list)) else o
        vf = vf.float()
        merge = getattr(self.vision.config, "spatial_merge_size", None) or getattr(self.vision.config, "merge_size", 2)
        counts = [int(g[0] * g[1] * g[2] // (merge * merge)) for g in grid.tolist()]
        assert sum(counts) == vf.shape[0], f"视觉 token 数对不上：{sum(counts)} vs {vf.shape[0]}"
        out, off = [], 0
        for c in counts:
            out.append(vf[off : off + c])
            off += c
        return out

    @torch.no_grad()
    def generate(self, prompts, images, max_new=24, items=None, bench=None, force_gen=False):
        pids = [self.tok(p)["input_ids"] for p in prompts]
        pad = max(len(x) for x in pids)
        attn = torch.tensor([[0] * (pad - len(x)) + [1] * len(x) for x in pids], device=self.dev)
        is_mc = [False if force_gen else bool(items and items[k].get("choices")) for k in range(len(prompts))]
        need_gen = (not items) or any(not m for m in is_mc)
        if self.mode == "teacher":
            texts = [topd.VISION_PREFIX + p for p in prompts]
            enc = self.proc(text=texts, images=images, return_tensors="pt", padding=True)
            enc = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
            Lp = enc["input_ids"].shape[1]
            lg = self.model(**enc).logits[:, -1, :].float() if any(is_mc) else None
            tails = []
            if need_gen:
                gkw = dict(**enc, max_new_tokens=max_new, do_sample=False,
                           no_repeat_ngram_size=0 if force_gen else 3, pad_token_id=self.pad_id, eos_token_id=self.stops)
                gen = self.model.generate(**gkw)
                tails = [g[Lp:] for g in gen]
        elif self.mode == "base":
            ids = torch.tensor([[self.pad_id] * (pad - len(x)) + x for x in pids], device=self.dev)
            lg = self.model(input_ids=ids, attention_mask=attn).logits[:, -1, :].float() if any(is_mc) else None
            tails = []
            if need_gen:
                gkw = dict(input_ids=ids, attention_mask=attn, max_new_tokens=max_new, do_sample=False,
                           no_repeat_ngram_size=0 if force_gen else 3, pad_token_id=self.pad_id, eos_token_id=self.stops)
                gen = self.model.generate(**gkw)
                tails = [g[pad:] for g in gen]
        elif self.mode in ("proj", "proj_pool32", "proj_fastv32"):
            pv, grid, lens = self._pix(images)
            feats = self._vision_feats(pv, grid, lens)
            emb = self.model.get_input_embeddings()
            rows, masks = [], []
            for k, ids in enumerate(pids):
                v = self.proj(feats[k]).to(self.model.dtype)
                if self.mode == "proj_pool32" and v.shape[0] > 32:
                    v = torch.nn.functional.adaptive_avg_pool1d(v.float().T.unsqueeze(0), 32).squeeze(0).T.to(v.dtype)
                elif self.mode == "proj_fastv32" and v.shape[0] > 32:
                    idx_top = v.float().norm(dim=-1).topk(32).indices.sort().values
                    v = v[idx_top]
                t = emb(torch.tensor(ids, device=self.dev))
                padt = torch.zeros(pad - len(ids), t.shape[-1], device=self.dev, dtype=t.dtype)
                rows.append(torch.cat([padt, v, t], dim=0))
                masks.append(torch.cat([torch.zeros(pad - len(ids), device=self.dev), torch.ones(v.shape[0] + len(ids), device=self.dev)]))
            n = max(r.shape[0] for r in rows)
            E = torch.zeros(len(rows), n, rows[0].shape[-1], device=self.dev, dtype=rows[0].dtype)
            M = torch.zeros(len(rows), n, device=self.dev, dtype=torch.long)
            for k, (r, m) in enumerate(zip(rows, masks)):
                E[k, n - r.shape[0]:] = r
                M[k, n - r.shape[0]:] = m.long()
            lg = self.model(inputs_embeds=E, attention_mask=M).logits[:, -1, :].float() if any(is_mc) else None
            tails = []
            if need_gen:
                gkw = dict(inputs_embeds=E, attention_mask=M, max_new_tokens=max_new, do_sample=False,
                           no_repeat_ngram_size=0 if force_gen else 3, pad_token_id=self.pad_id, eos_token_id=self.stops)
                gen = self.model.generate(**gkw)
                tails = [g for g in gen]
        else:
            pv, grid, lens = self._pix(images)
            if self.mode == "tower":
                feats = self._vision_feats(pv, grid, lens)
            else:
                vh, cl = self.front(pv.to(self.dev), grid, lens)
                feats = [vh[i, : cl[i]].float() for i in range(len(cl))]
            vis, vm = topd.pad_vis(feats, self.dev)
            ids = torch.tensor([[self.pad_id] * (pad - len(x)) + x for x in pids], device=self.dev)
            with rotation_context_with(self.registry, vis, vm, text_mask=attn):
                lg = self.model(input_ids=ids, attention_mask=attn).logits[:, -1, :].float() if any(is_mc) else None
                tails = []
                if need_gen:
                    gkw = dict(input_ids=ids, attention_mask=attn, max_new_tokens=max_new, do_sample=False,
                               no_repeat_ngram_size=0 if force_gen else 3, pad_token_id=self.pad_id, eos_token_id=self.stops)
                    gen = self.model.generate(**gkw)
                    tails = [g[pad:] for g in gen]
        out, stats = [], []
        stop_set = set(self.stops)
        for k in range(len(prompts)):
            if is_mc[k]:
                if bench == "hallusion" or items[k]["choices"] == ["Yes", "No"]:
                    cands = ["Yes", "No"]
                    c_lgs = [lg[k, self.yn_ids["Yes"]].item(), lg[k, self.yn_ids["No"]].item()]
                    out.append("Yes" if c_lgs[0] >= c_lgs[1] else "No")
                    stats.append({"cand_logits": c_lgs})
                else:
                    nc = min(len(items[k]["choices"]), len(self.let_ids))
                    cand = ["ABCDEFGH"[c] for c in range(nc)]
                    c_lgs = [lg[k, self.let_ids[l]].item() for l in cand]
                    out.append(cand[int(max(range(nc), key=lambda i: c_lgs[i]))])
                    stats.append({"cand_logits": c_lgs})
            else:
                raw_ids = [int(x) for x in tails[k].tolist()]
                has_eos = any(t in stop_set for t in raw_ids)
                cut_ids = topd.cut_at_stop(tails[k], stop_set)
                n_tok = len(cut_ids)
                ng3 = [tuple(cut_ids[j:j+3]) for j in range(max(0, n_tok - 2))]
                rep3 = int(len(ng3) >= 2 and len(set(ng3)) < len(ng3))
                out.append(self.tok.decode(cut_ids, skip_special_tokens=True))
                stats.append({"eos_stop": int(has_eos), "gen_tokens": n_tok, "rep3": rep3})
        self.last_gen_stats = stats
        return out
