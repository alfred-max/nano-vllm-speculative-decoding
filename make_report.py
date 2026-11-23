"""把 compare_*.json 合成 markdown 报告 + Pareto 图。"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load():
    return {json.load(open(f, encoding="utf-8"))["dataset"]: json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob("bench_results/compare_*.json"))}


def table(data):
    arms = ["base", "dflash", "dspark", "dspark-ad"]
    concs = sorted(next(iter(data["arms"]["base"].keys())), key=int) if False else \
        sorted(data["arms"]["base"].keys(), key=lambda x: int(x))
    lines = ["| 并发 | " + " | ".join(arms) + " | DFlash加速 | DSpark加速 |",
             "|---" * (len(arms) + 3) + "|"]
    for c in concs:
        base = data["arms"]["base"][c]["throughput"]
        row = [f"| {c} "]
        for a in arms:
            row.append(f"| {data['arms'][a][c]['throughput']:.0f} ")
        df = data["arms"]["dflash"][c]["throughput"] / base
        ds = data["arms"]["dspark"][c]["throughput"] / base
        row.append(f"| {df:.2f}× | {ds:.2f}× |")
        lines.append("".join(row))
    return "\n".join(lines)


def tau_table(data):
    arms = ["dflash", "dspark", "dspark-ad"]
    concs = sorted(data["arms"]["base"].keys(), key=lambda x: int(x))
    lines = ["| 并发 | " + " | ".join(arms) + " | mean_rows(adaptive) |",
             "|---" * (len(arms) + 2) + "|"]
    for c in concs:
        row = [f"| {c} "]
        for a in arms:
            t = data["arms"][a][c]["tau"]
            row.append(f"| {t:.2f} " if t else "| - ")
        mr = data["arms"]["dspark-ad"][c]["mean_rows"]
        row.append(f"| {mr:.2f} |" if mr else "| - |")
        lines.append("".join(row))
    return "\n".join(lines)


def plot(datasets):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    arms = ["base", "dflash", "dspark", "dspark-ad"]
    labels = {"base": "baseline", "dflash": "DFlash γ=7",
              "dspark": "DSpark-static", "dspark-ad": "DSpark-adaptive"}
    for ax, (name, data) in zip(axes, datasets.items()):
        concs = sorted(data["arms"]["base"].keys(), key=lambda x: int(x))
        base = [data["arms"]["base"][c]["throughput"] for c in concs]
        for a in arms:
            if a == "base":
                continue
            ys = [data["arms"][a][c]["throughput"] / b for c, b in zip(concs, base)]
            ax.plot([int(c) for c in concs], ys, marker="o", label=labels[a])
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrency")
        ax.set_ylabel("speedup vs baseline")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("bench_results/pareto.png", dpi=150)
    print("saved -> bench_results/pareto.png")


def main():
    ds = load()
    parts = ["# 投机解码对比实验（DFlash vs DSpark）\n"]
    for name, data in ds.items():
        parts.append(f"## {name}（γ={data['gamma']}，max_tokens={data['max_tokens']}）\n")
        parts.append("### 吞吐（tok/s）与加速比\n")
        parts.append(table(data) + "\n")
        parts.append("### 接受长度 τ 与验证行数\n")
        parts.append(tau_table(data) + "\n")
    open("report.md", "w", encoding="utf-8").write("\n".join(parts))
    print("saved -> report.md")
    plot(ds)


if __name__ == "__main__":
    main()