"""Tokenizer wrapper.

HAGI reuses an existing BPE tokenizer — we do not train one. Default is the
SmolLM2 tokenizer (~49K vocab), trained on FineWeb-Edu + code + math, which is
HAGI's exact data distribution and the right vocab size for a ~100M model
(embedding stays ~33% of params, not ~50% as with Llama-3/Qwen 128-151K vocabs).
"""

from __future__ import annotations

DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_VOCAB_SIZE = 49152


def load_tokenizer(name: str = DEFAULT_TOKENIZER):
    """Load a HF tokenizer. Imported lazily so the model package has no hard
    dependency on `transformers`."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
