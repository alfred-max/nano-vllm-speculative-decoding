"""投机解码 bench 基座：固定 prompt / 固定采样参数 / 结果与输出 token 落盘。

用法：
  python bench_spec.py --model ~/huggingface/Qwen3-0.6B --tag baseline --num-seqs 64 --temperature 0
  python bench_spec.py --model ... --prompt-file eval_datasets/gsm8k.jsonl --tag baseline-gsm8k
"""
import argparse
import json
import os
import subprocess
import time
from random import Random

from nanovllm import LLM, SamplingParams


def load_prompts(args):
    if args.prompt_file:
        prompts = []
        with open(args.prompt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompts.append(item.get("prompt") or item.get("question") or item["text"])
        return prompts[: args.num_seqs]
    rng = Random(args.seed)
    return [
        [rng.randint(0, 10000) for _ in range(rng.randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", default="baseline", help="这一臂的名字，也是输出文件名")
    parser.add_argument("--prompt-file", default=None, help="JSONL，每行含 prompt/question/text")
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out-dir", default="bench_results")
    parser.add_argument("--save-outputs", action="store_true", default=True)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--spec", default="none", choices=["none", "naive"])
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        spec_method=args.spec,
        num_speculative_tokens=args.num_speculative_tokens,
    )
    prompts = load_prompts(args)
    sampling_params = [
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True)
        for _ in prompts
    ]

    llm.generate(["warmup"], SamplingParams(max_tokens=8, ignore_eos=True))

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start

    num_out = sum(len(o["token_ids"]) for o in outputs)
    num_in = sum(len(p) if isinstance(p, list) else len(llm.tokenizer.encode(p)) for p in prompts)
    result = {
        "tag": args.tag,
        "model": args.model,
        "commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_seqs": len(prompts),
        "input_tokens": num_in,
        "output_tokens": num_out,
        "elapsed_s": round(elapsed, 3),
        "throughput_tok_s": round(num_out / elapsed, 2),
        "total_tok_s": round((num_in + num_out) / elapsed, 2),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enforce_eager": args.enforce_eager,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "spec": args.spec,
    }

    if getattr(llm, "spec_runtime", None) is not None:
        s = llm.spec_runtime.stats
        result["spec_stats"] = {
            "steps": s.steps,
            "seq_steps": s.seq_steps,
            "accept_len": round(s.accept_len, 3),
            "accept_rate": round(s.accept_rate, 3),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_outputs:
        outputs_path = os.path.join(args.out_dir, f"{args.tag}.outputs.json")
        with open(outputs_path, "w", encoding="utf-8") as f:
            json.dump([o["token_ids"] for o in outputs], f)
        result["outputs_file"] = outputs_path

    result_path = os.path.join(args.out_dir, f"{args.tag}.json")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()