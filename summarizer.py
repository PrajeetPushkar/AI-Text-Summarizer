# summarizer.py
"""
Quiet abstractive summarizer — prints ONLY the final summary (no warnings/progress).
Small, single-file.
"""
import os, subprocess, sys, argparse, warnings

# -------------- auto-install (keeps your original behavior) --------------
def auto_install(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ("transformers","sentencepiece","torch","nltk"):
    auto_install(p)

# -------------- silence/disable noisy output --------------
# Disable huggingface symlink warning and progress bars
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Reduce transformers/huggingface logging and Python warnings
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import logging, nltk
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# -------------- summarization logic (hierarchical for long inputs) --------------
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize
from transformers import pipeline

def chunk_sentences(sents, max_chars=1200):
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        l = len(s)
        if cur and cur_len + l > max_chars:
            chunks.append(" ".join(cur)); cur, cur_len = [], 0
        cur.append(s); cur_len += l
    if cur: chunks.append(" ".join(cur))
    return chunks

def summarize_text(text, model_name="t5-small", max_new_tokens=120, min_length=30, chunk_chars=1200, device=-1):
    sents = sent_tokenize(text)
    if not sents:
        return ""
    chunks = chunk_sentences(sents, max_chars=chunk_chars)

    # create pipeline quietly
    summarizer = pipeline("summarization", model=model_name, device=device)

    small_summaries = []
    for c in chunks:
        # use only max_new_tokens to avoid transformer warnings
        out = summarizer(c, min_length=min_length, max_new_tokens=max_new_tokens, truncation=True)[0]["summary_text"]
        small_summaries.append(out.strip())

    if len(small_summaries) == 1:
        return small_summaries[0]
    merged = " ".join(small_summaries)
    final = summarizer(merged, min_length=min_length, max_new_tokens=max_new_tokens, truncation=True)[0]["summary_text"]
    return final.strip()

# -------------- CLI --------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quiet abstractive summarizer (prints only summary)")
    p.add_argument("input", help="Path to input .txt file")
    p.add_argument("--model", default="google/pegasus-large", help="model")
    p.add_argument("--min_length", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=400,
                   help="controls how many tokens the model may output (use instead of max_length)")
    p.add_argument("--chunk_chars", type=int, default=1200)
    p.add_argument("--use_gpu", action="store_true")
    args = p.parse_args()

    device = 0 if args.use_gpu else -1
    with open(args.input, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        sys.exit(0)

    # Generate and print only the summary (no extra logging)
    summary = summarize_text(txt, model_name=args.model,
                             max_new_tokens=args.max_new_tokens,
                             min_length=args.min_length,
                             chunk_chars=args.chunk_chars,
                             device=device)
    # Final: print only the summary (clean output)
    print("===SUMMARY===")
    print(summary)
