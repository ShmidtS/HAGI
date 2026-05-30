"""HAGI adapter for EleutherAI lm-evaluation-harness.

Registers HAGI as an `lm_eval` model so standard benchmarks (gsm8k, arc_challenge,
boolq, hellaswag, winogrande, ...) run against it directly:

    lm_eval --model hagi \
        --model_args ckpt=checkpoints/gdr/step-00050000.pt,tokenizer=HuggingFaceTB/SmolLM2-135M \
        --tasks gsm8k,arc_challenge,boolq

Implements the two request types the harness needs:
  - loglikelihood     : multiple-choice / cloze tasks (ARC, BoolQ, HellaSwag, ...)
  - generate_until    : free-generation tasks (GSM8K)

Targets recent lm-eval (>=0.4). The harness API evolves; if a version mismatch
occurs, adjust the method signatures per `lm_eval/api/model.py`. Import is lazy
so the model package never hard-depends on lm-eval.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    _LM_EVAL_AVAILABLE = True
except ImportError:  # allow import without lm-eval installed
    _LM_EVAL_AVAILABLE = False

    class LM:  # minimal stub so the module imports
        pass

    def register_model(*_names):
        def deco(cls):
            return cls

        return deco


from prototype.data.tokenizer import DEFAULT_TOKENIZER, load_tokenizer
from prototype.training.loop import load_checkpoint


@register_model("hagi")
class HAGILMEval(LM):
    def __init__(self, ckpt: str, tokenizer: str = DEFAULT_TOKENIZER,
                 device: str = "cuda", max_length: int = 4096, **kwargs):
        super().__init__()
        if not _LM_EVAL_AVAILABLE:
            raise ImportError("lm-eval not installed. `pip install lm-eval`.")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.tokenizer = load_tokenizer(tokenizer)

        model, _ = load_checkpoint(ckpt, device=self.device)
        self.model = model.eval()

    def tok_encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s)

    @torch.no_grad()
    def loglikelihood(self, requests):
        """Each request: (context_str, continuation_str).
        Returns list of (sum_logprob, is_greedy)."""
        out = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            full = (ctx_ids + cont_ids)[-self.max_length:]
            x = torch.tensor([full[:-1]], device=self.device)
            logits = self.model(x)  # [1, T, V]
            logprobs = F.log_softmax(logits.float(), dim=-1)[0]

            n_cont = len(cont_ids)
            # Continuation occupies the last n_cont prediction positions.
            cont_logprobs = logprobs[-n_cont:]
            targets = torch.tensor(full[-n_cont:], device=self.device)
            tok_lp = cont_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            greedy = bool((cont_logprobs.argmax(-1) == targets).all().item())
            out.append((float(tok_lp.sum().item()), greedy))
        return out

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        out = []
        for req in requests:
            (text,) = req.args
            ids = self.tok_encode(text)[: self.max_length]
            if len(ids) < 2:
                out.append(0.0)
                continue
            x = torch.tensor([ids[:-1]], device=self.device)
            logits = self.model(x)
            logprobs = F.log_softmax(logits.float(), dim=-1)[0]
            targets = torch.tensor(ids[1:], device=self.device)
            tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            out.append(float(tok_lp.sum().item()))
        return out

    @torch.no_grad()
    def generate_until(self, requests):
        """Greedy generation until a stop string or max_gen_toks."""
        out = []
        for req in requests:
            context, gen_kwargs = req.args
            stops = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            max_gen = gen_kwargs.get("max_gen_toks", 256) if isinstance(gen_kwargs, dict) else 256
            ids = self.tok_encode(context)[-self.max_length:]
            generated = []
            for _ in range(max_gen):
                x = torch.tensor([ids[-self.max_length:]], device=self.device)
                logits = self.model(x)
                nxt = int(logits[0, -1].argmax().item())
                ids.append(nxt)
                generated.append(nxt)
                text = self.tokenizer.decode(generated)
                if any(s in text for s in stops):
                    for s in stops:
                        if s in text:
                            text = text[: text.index(s)]
                    break
            else:
                text = self.tokenizer.decode(generated)
            out.append(text)
        return out
