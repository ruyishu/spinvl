"""叙事 2 的数据：图片 → teacher 同款 patch（已按 2×2 合并窗口排序）+ 对应的老师视觉特征。"""

import glob as _glob
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

B = "/data/ibu-bi-ai-share-disk"
TEACHER = f"{B}/zk/model/Qwen3-VL-8B-Instruct"


def parse_source(spec):
    """'<meta 或通配>|<manifest.jsonl>|<img_root>' → (meta_path, {id: abs_image_path})"""
    meta, manifest, root = spec.split("|")
    paths = {}
    for line in open(manifest):
        r = json.loads(line)
        paths[r["id"]] = os.path.join(root, r["image"])
    return meta, paths


class FrontDataset(Dataset):
    def __init__(self, items, vfeat, image_processor):
        """items: [(idx_in_vfeat, image_path)]"""
        self.items, self.vfeat, self.ip = items, vfeat, image_processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        vi, path = self.items[k]
        img = Image.open(path).convert("RGB")
        out = self.ip(images=[img], return_tensors="pt")
        pv = out["pixel_values"].to(torch.float16)
        grid = out["image_grid_thw"][0]
        return pv, grid, self.vfeat[vi]


def collate(batch):
    lens = [b[0].shape[0] for b in batch]
    L = max(lens)
    pv = torch.zeros(len(batch), L, batch[0][0].shape[1], dtype=torch.float16)
    for i, b in enumerate(batch):
        pv[i, : lens[i]] = b[0]
    grid = torch.stack([b[1] for b in batch])
    return pv, grid, lens, [b[2] for b in batch]


def make_items(load_data, specs):
    """合并多个来源：返回 (items, 每个来源的 vfeat 视图合并后的 MultiLazy 风格索引)。"""
    items, views = [], []
    for spec in specs:
        meta, paths = parse_source(spec)
        data = load_data(meta)
        views.append(data["vfeat"])
        base = sum(len(v) for v in views[:-1])
        for i, rid in enumerate(data["ids"]):
            if rid in paths and data["vlen"][i] > 0:
                items.append((base + i, paths[rid]))
    return items, views


class Joined:
    """把多个 vfeat 视图按顺序接成一个全局下标视图。"""

    def __init__(self, views):
        self.views = views
        self.cum = [0]
        for v in views:
            self.cum.append(self.cum[-1] + len(v))

    def __getitem__(self, i):
        for k in range(len(self.views)):
            if i < self.cum[k + 1]:
                return self.views[k][i - self.cum[k]]
        raise IndexError(i)

    def __len__(self):
        return self.cum[-1]
