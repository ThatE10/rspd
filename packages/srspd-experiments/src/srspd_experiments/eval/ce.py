"""Teacher-forced cross-entropy readouts on a causal LM.

* :func:`block_ce` -- mean next-token CE over a token block, windowed, every
  position scored (the held-out readout estimator);
* :func:`teacher_forced_ce` -- CE of a continuation given a prompt, split at
  the prompt boundary.
"""
from __future__ import annotations

import torch

#: Default window for :func:`block_ce`. The float32 ``log_softmax`` is
#: ``[window, vocab]``; at a 262k vocabulary that is ~0.5 GB per 512 tokens.
HELDOUT_WINDOW = 512


@torch.no_grad()
def teacher_forced_ce(model, full_ids, prompt_len):
    """Teacher-forced CE of ``full_ids`` under ``model``, split at the prompt.

    ``full_ids`` is the *baseline* trajectory: prompt tokens followed by the
    continuation the unedited model greedily produced. Scoring it under an
    edited model asks "how surprised is the edited model by what it used to
    say?" -- 0 for the unedited model by construction, growing as the edit bites.

    One forward gives every position, so this costs a single pass rather than
    one per token::

        position j predicts token j + 1

    so the continuation tokens, at absolute indices ``P … L-1``, are predicted
    by positions ``P-1 … L-2``. Hence ``cut = prompt_len - 1`` in the ``[L-1]``
    per-position NLL vector, and **not** ``prompt_len``: an off-by-one here
    silently scores the prompt's last token as continuation, which is exactly
    the token the chat template contributes and the model is most certain of.

    EOS stays in the scored span. It is part of what the baseline chose to emit,
    and an edit that stops the model terminating is a real behavioural change.

    Returns
    -------
    (ce_cont, n_cont, ce_prompt, cut)
        Mean continuation NLL in nats; the number of continuation tokens
        scored, ``L - prompt_len``; the mean NLL over the ``cut`` prompt
        positions (``nan`` when the prompt is a single token, which has no
        predecessor to be predicted from); and ``cut`` itself, reported so a
        caller can assert the split rather than trust it.
    """
    ids    = full_ids.unsqueeze(0).to(model.device)
    logits = model(input_ids=ids).logits              # one forward -> all positions
    lp     = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt    = ids[0, 1:]
    nll    = -lp[torch.arange(tgt.shape[0], device=tgt.device), tgt]
    cut    = prompt_len - 1        # position `cut` predicts the FIRST continuation token
    return (float(nll[cut:].mean()), int(nll.shape[0] - cut),
            float(nll[:cut].mean()) if cut > 0 else float("nan"), cut)


@torch.no_grad()
def block_ce(model, toks, window: int = HELDOUT_WINDOW):
    """Mean next-token CE over a held-out token block (lower = better).

    ``log_softmax`` of the logits in float32, gather the next token, mean;
    every position is scored (no prompt to skip).

    The block is cut into consecutive windows of ``window`` tokens, each scored
    independently, and the NLLs pooled by token count: at a 262k vocabulary the
    float32 ``log_softmax`` of a few thousand tokens is several GB of transient.

    Windowing costs context at each seam (the first token of window *k* is
    predicted from nothing before it, and ``n_windows - 1`` targets drop out),
    so this is a slightly pessimistic estimate of the single-forward value. It
    is the *same* estimator before and after every edit, which is what a delta
    needs. Pass ``window=0`` for the exact single-forward value.

    Returns
    -------
    (ce, n_tokens)
        ``(nan, 0)`` for a block too short to have a single target.
    """
    if toks.shape[-1] < 2:
        return float("nan"), 0
    ids = toks.to(model.device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    T = int(ids.shape[-1])
    step = T if not window else int(window)
    tot, n = 0.0, 0
    for s in range(0, T, step):
        chunk = ids[:, s:s + step]
        if chunk.shape[-1] < 2:
            continue
        logits = model(input_ids=chunk).logits
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = chunk[0, 1:]
        tot += float(-lp[torch.arange(tgt.shape[0], device=tgt.device), tgt].sum())
        n += int(tgt.shape[0])
        del logits, lp
    return (tot / n if n else float("nan")), n
