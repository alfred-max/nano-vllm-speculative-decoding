"""侦察一个投机权重目录：config.json、权重名/形状、参数量。

用法: python scripts/inspect_drafter.py ~/models/dflash_qwen3_8b_block7
"""
import glob
import json
import os
import sys

from safetensors import safe_open


def main():
    path = sys.argv[1]
    print("=" * 100)
    print("目录:", path)
    print("文件:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(path, "*"))))

    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        print("=" * 100)
        print("config.json:")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        print("architectures:", cfg.get("architectures"))

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    print("=" * 100)
    print(f"权重文件 {len(files)} 个")

    rows, total = [], 0
    for fp in files:
        with safe_open(fp, "pt", "cpu") as f:
            for k in f.keys():
                shape = tuple(f.get_slice(k).get_shape())
                n = 1
                for s in shape:
                    n *= s
                total += n
                rows.append((k, shape, n))

    rows.sort()
    print(f"{'name':70s} {'shape':28s} {'params':>12s}")
    for k, shape, n in rows:
        print(f"{k:70s} {str(shape):28s} {n:12,d}")

    print("=" * 100)
    print(f"总参数量: {total:,}  ({total * 2 / 2**30:.2f} GiB @bf16)")

    prefixes = {}
    for k, _, n in rows:
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + n
    print("顶层前缀参数量:")
    for p, n in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {p:40s} {n:12,d}")


if __name__ == "__main__":
    main()