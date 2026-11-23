"""对比矩阵：baseline / DFlash / DSpark / DSpark-adaptive。"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ARMS = [
    ("base", []),
    ("dflash", ["--spec", "dflash"]),
    ("dspark", ["--spec", "dspark"]),
    ("dspark-ad", ["--spec", "dspark", "--spec-adaptive"]),
]


def one_run(model, prompt_file, dataset, arm, extra, c, rep, gamma, max_tokens):
    tag = f"cmp-{dataset}-{arm}-c{c}-r{rep}"
    cmd = [sys.executable, "bench_spec.py",
           "--model", model, "--tag", tag,
           "--prompt-file", prompt_file, "--chat-template",
           "--num-seqs", str(c), "--max-tokens", str(max_tokens),
           "--temperature", "0", "--max-num-seqs", str(c)]
    if arm != "base":
        cmd += ["--num-speculative-tokens", str(gamma)]
        cmd += extra
        cmd += ["--draft-model", os.environ["DRAFT"] if "dflash" in arm else os.environ["DSPARK"]]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(f"bench_results/{tag}.json", encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--concurrency", default="1,4,8,16,32")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--gamma", type=int, default=7)
    args = p.parse_args()

    concs = [int(x) for x in args.concurrency.split(",")]
    out = {"dataset": args.dataset, "model": args.model,
           "max_tokens": args.max_tokens, "gamma": args.gamma, "arms": {}}

    for arm, extra in ARMS:
        out["arms"][arm] = {}
        for c in concs:
            runs = [one_run(args.model, args.prompt_file, args.dataset, arm, extra, c, r,
                            args.gamma, args.max_tokens) for r in range(args.repeats)]
            tps = [r["throughput_tok_s"] for r in runs]
            taus = [r["spec_stats"]["accept_len"] for r in runs if "spec_stats" in r]
            rows = [r["spec_stats"]["mean_rows"] for r in runs if "spec_stats" in r]
            steps = [r["spec_stats"]["steps"] for r in runs if "spec_stats" in r]
            out["arms"][arm][c] = {
                "throughput": statistics.median(tps),
                "throughput_all": tps,
                "tau": statistics.median(taus) if taus else None,
                "mean_rows": statistics.median(rows) if rows else None,
                "steps": statistics.median(steps) if steps else None,
            }
            print(f"{args.dataset:8s} {arm:10s} c={c:3d}  "
                  f"{out['arms'][arm][c]['throughput']:8.1f} tok/s  "
                  f"tau={out['arms'][arm][c]['tau']}")

    path = f"bench_results/compare_{args.dataset}.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()