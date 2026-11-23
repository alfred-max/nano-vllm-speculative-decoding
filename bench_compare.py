"""对比矩阵：baseline / DFlash / DSpark / DSpark-adaptive。

臂与并发交错执行（每个 rep 内先遍历所有臂），避免机器状态漂移带来的系统性偏差。
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ARMS = [
    ("base", [], None),
    ("dflash", ["--spec", "dflash"], "draft"),
    ("dspark", ["--spec", "dspark"], "dspark"),
    ("dspark-ad", ["--spec", "dspark", "--spec-adaptive"], "dspark"),
]


def one_run(args, arm, extra, dm, c, rep):
    tag = f"cmp-{args.dataset}-{arm}-c{c}-r{rep}"
    cmd = [sys.executable, "bench_spec.py",
           "--model", args.model, "--tag", tag,
           "--prompt-file", args.prompt_file, "--chat-template",
           "--num-seqs", str(c), "--max-tokens", str(args.max_tokens),
           "--temperature", "0", "--max-num-seqs", str(c)]
    if arm != "base":
        cmd += ["--num-speculative-tokens", str(args.gamma)] + extra + ["--draft-model", dm]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return None, proc.stderr[-2000:]
    try:
        with open(f"bench_results/{tag}.json", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"读取结果失败: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--draft-model", default=os.environ.get("DRAFT"))
    p.add_argument("--dspark-model", default=os.environ.get("DSPARK"))
    p.add_argument("--concurrency", default="1,4,8,16,32")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--gamma", type=int, default=7)
    args = p.parse_args()

    if not args.draft_model or not args.dspark_model:
        raise SystemExit("缺少 drafter 路径：--draft-model / --dspark-model（或 DRAFT / DSPARK 环境变量）")

    concs = [int(x) for x in args.concurrency.split(",")]
    out = {"dataset": args.dataset, "model": args.model, "max_tokens": args.max_tokens,
           "gamma": args.gamma, "repeats": args.repeats, "arms": {}}
    runs = {}

    # 关键：外层 rep，内层 arm，再内层 concurrency —— 四臂在时间上交错
    for rep in range(args.repeats):
        for arm, extra, which in ARMS:
            dm = args.draft_model if which == "draft" else args.dspark_model
            for c in concs:
                rec, err = one_run(args, arm, extra, dm, c, rep)
                if rec is None:
                    print(f"[fail] {arm} c={c} r={rep}\n{err}")
                    continue
                runs.setdefault((arm, c), []).append(rec)
                print(f"r{rep} {arm:10s} c={c:3d}  {rec['throughput_tok_s']:8.1f} tok/s")

    for arm, _, _ in ARMS:
        out["arms"][arm] = {}
        for c in concs:
            got = runs.get((arm, c), [])
            if not got:
                out["arms"][arm][c] = {"throughput": None, "tau": None,
                                       "mean_rows": None, "steps": None}
                continue
            ss = [r["spec_stats"] for r in got if "spec_stats" in r]
            out["arms"][arm][c] = {
                "throughput": statistics.median(r["throughput_tok_s"] for r in got),
                "throughput_all": [r["throughput_tok_s"] for r in got],
                "tau": statistics.median(s["accept_len"] for s in ss) if ss else None,
                "mean_rows": statistics.median(s["mean_rows"] for s in ss) if ss else None,
                "steps": statistics.median(s["steps"] for s in ss) if ss else None,
            }

    path = f"bench_results/compare_{args.dataset}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'并发':>4} " + " ".join(f"{a:>10}" for a, _, _ in ARMS))
    for c in concs:
        cells = []
        for arm, _, _ in ARMS:
            v = out["arms"][arm][c]["throughput"]
            cells.append(f"{v:>10.1f}" if v else f"{'-':>10}")
        print(f"{c:>4} " + " ".join(cells))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()