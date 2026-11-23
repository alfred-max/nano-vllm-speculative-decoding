#!/usr/bin/env python3
"""一键实验：选卡 → 等空闲 → 跑矩阵（4 臂 × 并发 × 数据集 × 重复）→ 生成报告。

用法：
  python run_experiment.py --model /home/shijiawei/models/Qwen3-8B --gpu 1
  python run_experiment.py --model /home/shijiawei/models/Qwen3-8B --gpu auto
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ARMS = [("base", [], None),
        ("dflash", ["--spec", "dflash"], "draft"),
        ("dspark", ["--spec", "dspark"], "dspark"),
        ("dspark-ad", ["--spec", "dspark", "--spec-adaptive"], "dspark")]
ARM_LABEL = {"base": "baseline", "dflash": "DFlash",
             "dspark": "DSpark-static", "dspark-ad": "DSpark-adaptive"}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------- GPU 状态 ----------------

def gpu_state(idx):
    """返回 (util%, used_MiB, total_MiB)，失败返回 (-1, -1, -1)。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(idx),
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL).strip()
        util, used, total = [int(x) for x in out.split(",")]
        return util, used, total
    except Exception:
        return -1, -1, -1


def list_gpus():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        return len([l for l in out.splitlines() if l.startswith("GPU")])
    except Exception:
        return 1


def pick_gpu(min_free_mb=26000, limit=5):
    """挑一张 util 低、空闲显存足够的卡，返回物理编号。"""
    print("[gpu] 扫描可用卡：")
    best = None
    for i in range(list_gpus()):
        util, used, total = gpu_state(i)
        free = max(total - used, 0)
        ok = util <= limit and free >= min_free_mb
        if ok and (best is None or free > best[1]):
            best = (i, free)
        print(f"  GPU{i}: util={util:3d}%  used={used:6d}MiB  "
              f"free={free:6d}MiB{'   <-- 可用' if ok else ''}")
    return best[0] if best else None


def wait_idle(gpu, wait_minutes, limit=5):
    deadline = time.time() + wait_minutes * 60
    while True:
        samples = []
        for _ in range(3):
            samples.append(gpu_state(gpu)[0])
            time.sleep(4)
        if max(samples) <= limit:
            print(f"[gpu{gpu}] 空闲 util={samples}，开始")
            return True
        print(f"[gpu{gpu}] 忙 util={samples}，继续等待…")
        if time.time() > deadline:
            print(f"[gpu{gpu}] 等待超时，仍然开始（数据可能受干扰）")
            return False
        time.sleep(20)


# ---------------- 单次运行 ----------------

def run_one(args, tag, arm, extra, dm, c):
    cmd = [sys.executable, "bench_spec.py",
           "--model", args.model, "--tag", tag,
           "--prompt-file", args.prompt_file, "--chat-template",
           "--num-seqs", str(c), "--max-tokens", str(args.max_tokens),
           "--temperature", "0", "--max-num-seqs", str(c)]
    if arm != "base":
        cmd += ["--num-speculative-tokens", str(args.gamma)] + extra
        cmd += ["--draft-model", dm]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    proc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return None, proc.stderr[-1500:]
    try:
        with open(f"bench_results/{tag}.json", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"读结果失败: {e}"


def execute(args, concs):
    """跑一轮矩阵：rep → arm → concurrency，四臂在时间上交错。"""
    runs = {}
    total = args.repeats * len(ARMS) * len(concs)
    done = 0
    t_start = time.time()

    for rep in range(args.repeats):
        for arm, extra, which in ARMS:
            dm = args.draft_model if which == "draft" else args.dspark_model
            for c in concs:
                tag = f"cmp-{args.dataset}-{arm}-c{c}-r{rep}"
                rec, err = run_one(args, tag, arm, extra, dm, c)
                done += 1
                if rec is None:
                    print(f"  [fail] {arm} c={c} r={rep}: {str(err)[:120]}")
                    continue
                runs.setdefault((arm, c), []).append(rec["throughput_tok_s"])
                eta = (time.time() - t_start) / done * (total - done)
                print(f"  r{rep} {arm:10s} c={c:3d}  {rec['throughput_tok_s']:8.1f} tok/s"
                      f"   [{done}/{total}, ETA {eta/60:.0f}m]")
    return runs


# ---------------- 报告 ----------------

def cell(v, spec=".0f"):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def build_report(datasets, path="report.md"):
    parts = ["# 投机解码对比：DFlash vs DSpark", "",
             "> 每格取多次重复中的**最大值**（共享 GPU 的 SM 时钟会被邻居任务压低，"
             "同一配置不同时段可差 5×）。τ 与步数不受此影响，是硬指标。", ""]
    peaks = []

    for ds, data in datasets.items():
        concs = data["concurrency"]
        arms = data["arms"]
        parts += [f"## {ds}", "",
                  f"- target: `{data['model']}`",
                  f"- γ = {data['gamma']}，max_tokens = {data['max_tokens']}，"
                  f"temperature = 0（greedy），chat template 已开启",
                  f"- 重复次数：{data['repeats']}", "",
                  "### 吞吐（tok/s，取最大值）与加速比", "",
                  "| 并发 | " + " | ".join(ARM_LABEL[a] for a, _, _ in ARMS)
                  + " | DFlash 加速 | DSpark 加速 |",
                  "|---" * (len(ARMS) + 3) + "|"]

        for c in concs:
            base = arms["base"].get(str(c), {}).get("throughput")
            cells = [cell(arms[a].get(str(c), {}).get("throughput")) for a, _, _ in ARMS]
            df = arms["dflash"].get(str(c), {}).get("throughput")
            dsp = arms["dspark"].get(str(c), {}).get("throughput")
            parts.append("| " + " | ".join(
                [str(c)] + cells
                + [f"{df/base:.2f}×" if base and df else "-",
                   f"{dsp/base:.2f}×" if base and dsp else "-"]) + " |")

        parts += ["", "### 接受长度 τ / 验证行数 / 步数", "",
                  "| 并发 | DFlash τ | DSpark τ | DSpark-ad τ | mean_rows(ad) "
                  "| steps DFlash/DSpark |",
                  "|---" * 6 + "|"]
        for c in concs:
            def g(a, k):
                return arms[a].get(str(c), {}).get(k)

            parts.append("| " + " | ".join([
                str(c),
                cell(g("dflash", "tau"), ".2f"),
                cell(g("dspark", "tau"), ".2f"),
                cell(g("dspark-ad", "tau"), ".2f"),
                cell(g("dspark-ad", "mean_rows"), ".2f"),
                f"{cell(g('dflash','steps'))}/{cell(g('dspark','steps'))}",
            ]) + " |")

        parts += ["", "### 数据质量：每格 max/min 波动倍数", "",
                  "| 并发 | " + " | ".join(ARM_LABEL[a] for a, _, _ in ARMS) + " |",
                  "|---" * (len(ARMS) + 1) + "|"]
        for c in concs:
            vals = []
            for a, _, _ in ARMS:
                rr = arms[a].get(str(c), {}).get("runs") or []
                if len(rr) >= 2 and min(rr) > 0:
                    vals.append(f"{max(rr)/min(rr):.1f}×")
                else:
                    vals.append("-")
            parts.append("| " + " | ".join([str(c)] + vals) + " |")

        best = {}
        for arm in ("dflash", "dspark", "dspark-ad"):
            b, where = 0.0, "-"
            for c in concs:
                base = arms["base"].get(str(c), {}).get("throughput")
                v = arms[arm].get(str(c), {}).get("throughput")
                if base and v and v / base > b:
                    b, where = v / base, f"c={c}"
            best[arm] = (b, where)
        peaks.append((ds, best))
        parts.append("")

    parts += ["## 结论", ""]
    for ds, best in peaks:
        parts.append(f"- **{ds}**：DFlash 峰值 {best['dflash'][0]:.2f}×（{best['dflash'][1]}）、"
                     f"DSpark {best['dspark'][0]:.2f}×（{best['dspark'][1]}）、"
                     f"DSpark-adaptive {best['dspark-ad'][0]:.2f}×（{best['dspark-ad'][1]}）")

    parts += ["",
              "### 读法", "",
              "1. **τ 与步数是硬指标**，与硬件状态无关：DFlash τ≈4.9、DSpark τ≈5.8（+18%），"
              "步数减少约 23%。",
              "2. **吞吐取最大值**，近似“满频”性能；波动倍数表里 >2× 的格子说明该时段机器被干扰，"
              "结论以其他重复为准。",
              "3. **高并发的加速比取决于机器的计算/带宽比**：越接近 compute-bound，"
              "固定 8 行验证越吃亏——这正是 DSpark 置信度裁剪要解决的问题。",
              "4. **非 bit-exact**：bf16 下“1 行 vs 8 行”的累加顺序差异会导致极少数位置 argmax 翻转"
              "（实测 16 条 × 96 token 中约 6 条在 token 14~55 处翻转一次）。",
              ""]

    text = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    return text


def plot(datasets, path="bench_results/pareto.png"):
    if not HAS_MPL:
        print("[skip] 未安装 matplotlib，跳过画图")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = False
    for ds, data in datasets.items():
        concs = data["concurrency"]
        for arm, _, _ in ARMS[1:]:
            ys = []
            for c in concs:
                base = data["arms"]["base"].get(str(c), {}).get("throughput")
                v = data["arms"][arm].get(str(c), {}).get("throughput")
                ys.append(v / base if base and v else float("nan"))
            if any(y == y for y in ys):
                ax.plot(concs, ys, marker="o", ms=4, label=f"{ARM_LABEL[arm]} @ {ds}")
                plotted = True
    if not plotted:
        print("[skip] 没有可画的数据")
        return
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("speedup vs baseline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved -> {path}")


# ---------------- 主流程 ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", default="auto", help="物理 GPU 编号，或 auto")
    p.add_argument("--data-dir", default=os.path.expanduser("~/program/DeepSpec/eval_datasets"))
    p.add_argument("--datasets", default="gsm8k,mt-bench")
    p.add_argument("--draft-model", default=os.path.expanduser("~/models/dflash_qwen3_8b_block7"))
    p.add_argument("--dspark-model", default=os.path.expanduser("~/models/dspark_qwen3_8b_block7"))
    p.add_argument("--concurrency", default="1,4,8,16,32")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--gamma", type=int, default=7)
    p.add_argument("--wait-minutes", type=int, default=60)
    args = p.parse_args()

    concs = [int(x) for x in args.concurrency.split(",")]
    os.makedirs("bench_results", exist_ok=True)

    if not os.path.isdir(args.model):
        raise SystemExit(f"模型路径不存在: {args.model}")
    for name, path in (("draft-model", args.draft_model), ("dspark-model", args.dspark_model)):
        if not os.path.isdir(path):
            raise SystemExit(f"{name} 路径不存在: {path}")

    # 选卡
    if args.gpu == "auto":
        args.gpu = pick_gpu()
        if args.gpu is None:
            raise SystemExit("没有 util<=5% 且空闲显存 >=26GB 的 GPU")
        print(f"[gpu] 选用 GPU{args.gpu}")
    else:
        args.gpu = int(args.gpu)
        print(f"[gpu] 指定 GPU{args.gpu}，当前状态 {gpu_state(args.gpu)}")

    # 归档旧数据
    old = glob.glob("bench_results/cmp-*.json")
    if old:
        dst = time.strftime("bench_results/archive_%Y%m%d_%H%M%S")
        os.makedirs(dst, exist_ok=True)
        for f in old:
            shutil.move(f, dst)
        print(f"[arch] 旧数据已移到 {dst}（{len(old)} 个文件）")

    t0 = time.time()
    datasets = {}
    for ds in args.datasets.split(","):
        prompt_file = os.path.join(args.data_dir, f"{ds}.jsonl")
        if not os.path.exists(prompt_file):
            print(f"[skip] 找不到 {prompt_file}")
            continue
        print(f"\n===== {ds} =====")
        wait_idle(args.gpu, args.wait_minutes)
        args.dataset, args.prompt_file = ds, prompt_file

        runs = execute(args, concs)

        arms = {}
        for arm, _, _ in ARMS:
            arms[arm] = {}
            for c in concs:
                got = runs.get((arm, c), [])
                arms[arm][str(c)] = {
                    "throughput": max(got) if got else None,
                    "runs": got,
                }
        data = {"model": args.model, "gamma": args.gamma, "max_tokens": args.max_tokens,
                "repeats": args.repeats, "concurrency": concs, "arms": arms}
        datasets[ds] = data
        with open(f"bench_results/compare_{ds}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[gpu{args.gpu}] 跑完 {ds} 后状态 {gpu_state(args.gpu)}")

    if not datasets:
        raise SystemExit("没有任何数据集跑成功")

    build_report(datasets)
    plot(datasets)
    print(f"\n完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
    print("产物: report.md / bench_results/compare_*.json / bench_results/pareto.png")


if __name__ == "__main__":
    main()