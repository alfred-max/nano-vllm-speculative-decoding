"""合成对比报告：优先读 compare_*.json，没有就从 cmp-*.json 重建。"""
import glob
import json
import os
import re
import statistics

ARMS = [("base", "baseline"), ("dflash", "DFlash"),
        ("dspark", "DSpark-static"), ("dspark-ad", "DSpark-adaptive")]
ARM_NAMES = [a for a, _ in ARMS]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TAG_RE = re.compile(r"^cmp-(?P<ds>.+?)-(?P<arm>base|dflash|dspark|dspark-ad)-c(?P<c>\d+)-r(?P<r>\d+)$")


# ---------- 数据加载 ----------

def load_compare():
    out = {}
    for path in sorted(glob.glob("bench_results/compare_*.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        out[d["dataset"]] = d
    return out


def _arm_of(name: str):
    stem = name[:-5] if name.endswith(".json") else name
    for arm in ("dspark-ad", "dspark", "dflash", "base"):
        if f"-{arm}-c" in stem:
            return arm
    return None


def _dataset_of(rec) -> str:
    base = os.path.basename(rec.get("prompt_file") or "")
    if base.endswith(".jsonl"):
        base = base[:-6]
    return base or "unknown"


def load_from_tags():
    groups = {}
    for path in glob.glob("bench_results/cmp-*.json"):
        if path.endswith(".outputs.json"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        if "throughput_tok_s" not in rec:
            continue
        arm = _arm_of(os.path.basename(path))
        if arm is None:
            continue
        ds = _dataset_of(rec)
        c = rec.get("num_seqs")
        if c is None:
            continue
        groups.setdefault((ds, arm, int(c)), []).append(rec)

    if not groups:
        return {}

    datasets = {}
    for (ds, arm, c), runs in sorted(groups.items()):
        d = datasets.setdefault(ds, {
            "dataset": ds,
            "model": runs[0].get("model", "?"),
            "max_tokens": runs[0].get("max_tokens"),
            "gamma": runs[0].get("num_speculative_tokens", 7) or 7,
            "arms": {},
        })
        ss = [r["spec_stats"] for r in runs if "spec_stats" in r]
        d["arms"].setdefault(arm, {})[str(c)] = {
            "throughput": statistics.median(r["throughput_tok_s"] for r in runs),
            "tau": statistics.median(s["accept_len"] for s in ss) if ss else None,
            "mean_rows": statistics.median(s["mean_rows"] for s in ss) if ss else None,
            "steps": statistics.median(s["steps"] for s in ss) if ss else None,
        }
    return datasets


# ---------- 取数（全部兜底） ----------

def get(d, arm, c, key):
    return d.get("arms", {}).get(arm, {}).get(str(c), {}).get(key)


def concs_of(d):
    keys = set()
    for arm in ARM_NAMES:
        keys |= {int(c) for c in d.get("arms", {}).get(arm, {})}
    return sorted(keys)


def fmt(v, spec=".0f"):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


# ---------- 表格 ----------

def throughput_table(d):
    lines = ["| 并发 | " + " | ".join(lb for _, lb in ARMS) + " | DFlash 加速 | DSpark 加速 |",
             "|---" * (len(ARMS) + 3) + "|"]
    for c in concs_of(d):
        base = get(d, "base", c, "throughput")
        cells = [fmt(get(d, arm, c, "throughput")) for arm in ARM_NAMES]
        df = get(d, "dflash", c, "throughput")
        ds = get(d, "dspark", c, "throughput")
        lines.append("| " + " | ".join(
            [str(c)] + cells
            + [f"{df/base:.2f}×" if base and df else "-",
               f"{ds/base:.2f}×" if base and ds else "-"]) + " |")
    return "\n".join(lines)


def quality_table(d):
    lines = ["| 并发 | DFlash τ | DSpark τ | DSpark-ad τ | mean_rows(ad) | steps DFlash/DSpark |",
             "|---" * 6 + "|"]
    for c in concs_of(d):
        lines.append("| " + " | ".join([
            str(c),
            fmt(get(d, "dflash", c, "tau"), ".2f"),
            fmt(get(d, "dspark", c, "tau"), ".2f"),
            fmt(get(d, "dspark-ad", c, "tau"), ".2f"),
            fmt(get(d, "dspark-ad", c, "mean_rows"), ".2f"),
            f"{fmt(get(d, 'dflash', c, 'steps'))}/{fmt(get(d, 'dspark', c, 'steps'))}",
        ]) + " |")
    return "\n".join(lines)


def best_speedup(datasets, arm):
    best, where = 0.0, "-"
    for name, d in datasets.items():
        for c in concs_of(d):
            b, v = get(d, "base", c, "throughput"), get(d, arm, c, "throughput")
            if b and v and v / b > best:
                best, where = v / b, f"{name} c={c}"
    return best, where


# ---------- 画图 ----------

def plot(datasets):
    finite = 0
    for d in datasets.values():
        for c in concs_of(d):
            for arm in ARM_NAMES[1:]:
                b, v = get(d, "base", c, "throughput"), get(d, arm, c, "throughput")
                if b and v:
                    finite += 1
    if finite == 0:
        print("[skip] 没有可画的数据（所有臂都是空的）")
        return
    if not HAS_MPL:
        print("[skip] 未安装 matplotlib，跳过画图（pip install matplotlib）")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, d in datasets.items():
        concs = concs_of(d)
        for arm, label in ARMS[1:]:
            ys = []
            for c in concs:
                b, v = get(d, "base", c, "throughput"), get(d, arm, c, "throughput")
                ys.append(v / b if b and v else float("nan"))
            ax.plot(concs, ys, marker="o", ms=4, label=f"{label} @ {name}")
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("speedup vs baseline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("bench_results/pareto.png", dpi=150)
    print("saved -> bench_results/pareto.png")


# ---------- 主流程 ----------

def main():
    datasets = load_compare()
    src = "compare_*.json"
    if not datasets:
        datasets = load_from_tags()
        src = "cmp-*.json（重建）"
    if not datasets:
        raise SystemExit("bench_results/ 下既没有 compare_*.json 也没有 cmp-*.json")

    parts = ["# 投机解码对比：DFlash vs DSpark", "",
             f"数据来源：`bench_results/{src}`", ""]
    for name, d in datasets.items():
        parts += [f"## {name}", "",
                  f"- target: `{d.get('model', '?')}`",
                  f"- γ = {d.get('gamma', 7)}，max_tokens = {d.get('max_tokens')}，"
                  f"temperature = 0（greedy），chat template 已开启", "",
                  "### 吞吐（tok/s）与加速比", "", throughput_table(d), "",
                  "### 接受长度 τ 与验证行数", "", quality_table(d), ""]

    df_b, df_w = best_speedup(datasets, "dflash")
    ds_b, ds_w = best_speedup(datasets, "dspark")
    da_b, da_w = best_speedup(datasets, "dspark-ad")
    parts += ["## 结论", "",
              f"- DFlash 峰值加速 **{df_b:.2f}×**（{df_w}）",
              f"- DSpark-static 峰值加速 **{ds_b:.2f}×**（{ds_w}）",
              f"- DSpark-adaptive 峰值加速 **{da_b:.2f}×**（{da_w}）", "",
              "### 局限", "",
              "1. 高并发下每步固定开销（读权重 + drafter 前向）不摊薄，验证行数却随并发增长。",
              "2. 置信度裁剪在高接受率下不触发（GSM8K 尾部存活率仍 >0.6）；低接受场景才裁到 4~5 行。",
              "3. drafter 的 propose + ingest 占每步 30~50%，且随 bs 线性增长（Python 逐序列循环）。",
              "4. 非 bit-exact：bf16 下 1 行 vs 8 行的累加顺序差异会导致极少数位置 argmax 翻转。",
              ""]

    text = "\n".join(parts)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("saved -> report.md")
    plot(datasets)


if __name__ == "__main__":
    main()