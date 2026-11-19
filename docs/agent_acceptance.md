# Fresh-agent acceptance check

`AGENTS.md` is accepted when an agent that has read **only `AGENTS.md`** (not the source,
not the other docs) can, given a model, inspect it, erase with a `mu_per_token`, report the
leaks, switch the dose, restore the model and export the edit.

## The check

1. Start an agent with an empty context, the repository checkout and an environment that
   has `torch` and `transformers` but not necessarily an installed `rspd`. Give it
   `AGENTS.md` and this task, and forbid reading anything under `src/` or `docs/`:

   > Using a tiny random-weight GPT-2, write and run a script that inspects the model,
   > erases with a `mu_per_token`, prints the leaks of every edited matrix, switches to a
   > second dose, restores the model bit for bit, and exports the edited checkpoint.

2. Let it write the script in a scratch directory outside the repository and run it.
3. The check passes when the script runs to completion and:
   * `inspect_model` reports the matrices it then edits;
   * the printed leaks are `L_X_rel` and `L_Y_rel` for **every** edited matrix;
   * the second dose is applied with `report.apply_dose` (no second `erase`), and a larger
     `mu_per_token` gives a smaller `L_X_rel`;
   * after `rspd.api.restore`, every parameter equals its pristine value (`torch.equal`);
   * the exported checkpoint reloads with `transformers` and equals the edited weights;
   * the agent never passes `mu` or `mu_raw` as a dose and never writes anything but
     `W − M` into the model.
4. Every point where the agent had to guess, or read the source, is a documentation gap:
   fix `AGENTS.md` (not the code) and run the check again. A defect of the code's usability
   is reported, not patched as part of the check.

The same steps work with a real model name in place of the tiny GPT-2 (`AGENTS.md`,
"Standard workflow" shows the loading lines), at the cost of a download.

## Record

**2026-11-11.** Performed by the documentation author, writing the script from `AGENTS.md`
alone (the author had read the source, so this is a weaker test than a truly fresh agent;
repeat it with one).

* First run: `ModuleNotFoundError: No module named 'rspd'`. `AGENTS.md` said nothing about
  installation or running from a checkout. Fixed: a **Setup** section (git install, or
  `PYTHONPATH` from a checkout, and `python -m rspd.cli` when the console script is not
  installed).
* Second run: passed. Inspect (12 matrices, preset `gpt2`), erase at `mu_per_token` 0.5
  with a grid `[0.5, 2.0]`, leaks printed for all 12 matrices, `apply_dose(2.0)` lowered the
  median `c_fc` `L_X_rel` from 0.615 to 0.324, `restore` was bit for bit, the export reloaded
  equal to the edited weights, `close()` left the model pristine, and `rspd report` read
  the saved JSON.
* Also added after the run: how to load a model by name and pass text with a tokenizer
  (the task statement in the tracking issue starts from a model name).

Observed but not blocking (code usability, reported rather than fixed): the full `γ`
spectrum needed by `suggest_mu_per_token` is not in the `EditReport`; `report.matrices` is
in solve order rather than layer order; the tiny model's LayerNorm-fed matrices produce one
"not numerically positive definite" warning per matrix and dose.
