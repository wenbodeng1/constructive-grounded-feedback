"""Token statistics of dumped reference files (needs `transformers`; run e.g.
with /opt/conda/envs/vllm-rollout/bin/python; CPU only, tokenizer files only)."""
import glob, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from ct_ref import estimate_tokens

def stats(v):
    v = sorted(v); n = len(v)
    return "min=%d p25=%d median=%d mean=%.0f p75=%d p90=%d max=%d" % (v[0], v[n//4], v[n//2], sum(v)/n, v[3*n//4], v[int(0.9*n)], v[-1])

d = sys.argv[1]
models = sys.argv[2:] or ["Qwen/Qwen3-VL-8B-Instruct", "NousResearch/Hermes-3-Llama-3.1-8B"]
files = sorted(f for f in glob.glob(os.path.join(d, "*.txt")) if not os.path.basename(f).startswith("_"))
texts = [open(f, encoding="utf-8").read() for f in files]
est = [estimate_tokens(t) for t in texts]
print(d, "files", len(files), "| chars", stats([len(t) for t in texts]), "| est", stats(est))
for m in models:
    tok = AutoTokenizer.from_pretrained(m)
    n = [len(tok(t)["input_ids"]) for t in texts]
    ratio = [a / b for a, b in zip(est, n)]
    print("%s tokens %s | chars/token %.2f | est/actual min %.3f max %.3f" % (
        os.path.basename(m), stats(n), sum(map(len, texts)) / sum(n), min(ratio), max(ratio)))
