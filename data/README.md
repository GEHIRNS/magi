# Data behind the paper

All values were measured by us on one workstation: an RTX 5070 with 12 GB, running Windows 11 and Ollama 0.40.3. Columns computed from other columns are marked INFERRED. Table and figure numbers refer to `paper/where-the-bits-go.pdf`.

The files contain numbers and task identifiers only: no problem text, prompts, model outputs or thinking traces.

**`results.csv`: the full benchmark, one row per configuration.**
- Evidence: MEASURED. `capability_density_halvings_per_GiB` is INFERRED.
- Backs Table 7, Figure 1, Figure 7 and Figure 8b–c.
- Setup: 12 suites, 2,313 tasks, thinking on (budget-forced to at most 4,096 thinking tokens), greedy decoding, `num_ctx` 32,768.
- `index_pct` is the unweighted mean of the 12 suite pass rates. It is not `tasks_passed / tasks_total`.
- `weights_GiB` is the weight tensor data the runtime loads. For Q4_K_M it is the file size, as in the paper.
- `footprint_GiB` is the resident size Ollama reports through `/api/ps`, text-only. It excludes the roughly 1.05 GiB CUDA context.
- `capability_density_halvings_per_GiB` = −log2(1 − index) / footprint.
- `decode_tok_per_s` = `decode_tokens` / `decode_time_s`, summed over all 2,313 tasks.
- `wall_time_h` is the wall time the harness recorded for each task, which includes prompt processing and decoding, summed over the 2,313 tasks, in hours (Section 5.5).
- The magi-beta-1 footprint was recorded from Ollama's rounded display ("3.3 GB"), so it is only accurate to about ±0.05 GiB. Its tensor data are 0.030 GiB larger than magi-1.0's, which suggests about 3.05 GiB.

**`suites.csv`: tasks passed per suite for the same six configurations.**
- Evidence: MEASURED.
- Backs Table 7 and Figure 6.
- Columns: `passed`, `total` and `pass_rate_pct` (%).
- Code suites are graded by executing tests: EvalPlus base and extended tests for HumanEval+ and MBPP+, and the problems' test cases for LiveCodeBench.
- The other suites are graded by the extracted answer.

**`tasks.csv`: per-task outcomes of the full benchmark, one row per task (2,313 rows).**
- Evidence: MEASURED. `*_loop` is INFERRED from the thinking trace by the rule below.
- Lets you recompute every per-suite pass rate, index, paired sign test and loop count in Section 5.
- `suite`, `task_id`: the harness's identifiers (see "Benchmark items" below).
- `screen_set`: `decisive`, `q2_only` or `control` for the 126 tasks of the behavioural screen, empty otherwise.
- For each configuration, three 0/1 columns with the prefix `bf16`, `q4km` (Unsloth Q4_K_M), `q2k` (bartowski Q2_K), `magi` (magi-1.0), `magi_q8kv` (magi-1.0 + 8-bit KV) or `beta` (magi-beta-1):
  - `_pass`: the graded answer passed.
  - `_loop`: the last 4,000 characters of the thinking trace compress under zlib (Python `zlib.compress`, default level 6) to less than 10% of their length. Traces shorter than 1,000 characters never count as loops. The rule has not been validated against hand labels.
  - `_budget_hit`: the model used the whole 4,096-token thinking budget and was forced to answer.

**`failures.csv`: failed tasks split by loops, per configuration.**
- Evidence: INFERRED from `tasks.csv`.
- Backs Figure 8a and the loop counts in Sections 5.2 and 5.5.
- `failed_with_loop` + `failed_without_loop` = `tasks_failed`. `loops_all_tasks` also counts loops in passed tasks. `think_budget_exhausted` counts tasks with `_budget_hit` = 1.

**`screen.csv`, `screen_tasks.csv`, `screen_pairs.csv`: the behavioural screen (126 tasks).**
- Evidence: MEASURED; loops and sign tests are INFERRED.
- Backs Table 6, Figure 4, Table 8 and the paired tests in Sections 5.2 and 6.
- The screen holds 86 `decisive` tasks (Q4_K_M solved, Q2_K failed: 31 HumanEval+, 30 LiveCodeBench, 25 MMLU), 10 `q2_only` tasks (the reverse) and 30 `control` tasks that both solved (10 per suite, drawn at random).
- `screen.csv`: one row per model with its footprint (blank where not measured), decisive solved and loops, Q2-only and controls solved, and totals over all 126 tasks. `imatrix` names the importance matrix the file was built with; `outcomes_from` says whether the outcomes come from a separate screen run or from the model's full benchmark run (same protocol).
- `screen_tasks.csv`: per-task 0/1 `_pass` and `_loop` columns for each model, with the prefix given in `screen.csv`. A blank `_pass` means the answer was not graded.
- `screen_pairs.csv`: paired comparisons on the decisive tasks (all 86 unless `decisive_tasks` says otherwise). `solves_gained` counts tasks the model solves and the reference does not, `solves_lost` the reverse; the loop columns count tasks on which only one side loops. p-values are exact two-sided sign tests.
- Model names follow Table 4 of the paper. "random L" is the random allocation with seed 3 at budget L. "magi-1.0 + q8 KV (8192 budget)" is paired against "magi-1.0 + q8 KV" at 4,096. "candidate d" and the FFN-gate variant are described in Section 6 and Table 4.
- "candidate d with magi-1.0's attention" (prefix `candidate_d_q4attn`) keeps candidate d's FFN reductions but stores the attention projections at magi-1.0's Q4_K. It was built with the own-trace matrix. The grader failed on its HumanEval+ answers, so its solved columns are blank; its loop counts are complete. Without HumanEval+ it solves 22 of the 55 other decisive tasks and 17 of the 20 other controls. `screen_pairs.csv` gives it two rows: loops on all 86 decisive tasks, and solves and loops on the 55 without HumanEval+.
- Two more rows complete the two-by-two comparison of Section 6, in which all four builds use the own-trace matrix: candidate d against magi-beta-1 (the FFN reductions with Q6_K attention) and candidate d against candidate d with magi-1.0's attention (Q6_K against Q4_K attention; loops only).
- magi-beta-1 has three rows:
  - "magi-beta-1 pre-release build" (prefix `beta_prerelease`): the build that was screened before the release. It has the same tensor types as the released file but was assembled from tensors quantized with Unsloth's importance matrix.
  - "magi-beta-1" (prefix `beta`): the released file. It was not screened separately; its outcomes come from its full benchmark run, and they fall one decisive task short of the gate (51 against the bar of 52).
  - "magi-beta-1 + q8 KV" (prefix `beta_q8kv`): the released file with the 8-bit cache. The grader failed on its HumanEval+ answers, so its solved counts are blank; its loop counts are complete.
- The FFN-gate variant, greedy L and random L were also built with Unsloth's importance matrix; magi-1.0, its cache and budget variants, candidate d and candidate d with magi-1.0's attention use the own-trace matrix.
- Repeatability: magi-1.0 and magi-1.0 + q8 KV each ran these 126 tasks twice, in the screen and in the full run. Thinking traces, answers and token counts were identical on all 126.

**`sensitivity.csv`: the sensitivity map (44 single-change mixtures and the 2 references).**
- Evidence: MEASURED. `J_per_GiB_saved_or_spent` is INFERRED.
- Backs Figure 2 (the `down_from_ceiling` rows) and the depth result in Section 4.2.
- Divergence is measured by teacher forcing (`llama-perplexity` with KL-divergence output): the percentage of positions where the quantized model's greedy next token differs from BF16's. There are two sets:
  - E-long: 8 thinking traces that Qwen3.5-4B (Unsloth Q8_0) wrote for LiveCodeBench problems dated before 2025-01-18, each with its problem statement, concatenated (about 121 KB) and cut into four windows of 8,192 tokens. The second half of each window is scored, with the first half as context: 16,384 scored tokens in all. "Depth 4,096–8,192" is therefore a position within a window, not within one trace. The traces are disjoint from the 32 used for the importance matrix.
  - E-chat: 8,192 scored tokens of the same model's answers to 30 everyday prompts.
- `J_pct` is the mean of the two sets.
- `down_from_ceiling`: one tensor class is lowered from the Q4_K ceiling to IQ2_XXS. For the embedding, it is lowered from Q6_K to Q4_K.
- `up_from_floor`: one class, or one quarter of the layers, is raised from the IQ2_XXS floor.
- `J_per_GiB_saved_or_spent` = |ΔJ| / |ΔGiB|. Sizes are exact tensor bytes, in GiB.
- `params_share_pct` is read from the magi-1.0 GGUF header.

**`allocation.csv`: greedy, random and uniform allocations against published files.**
- Evidence: MEASURED. `J_pct` is INFERRED (the mean of the two measured columns).
- Backs Table 5, Figure 3 and the comparison of magi-beta-1 with magi-1.0 in Section 7.
- Budgets S and L match the file sizes of Unsloth's UD-IQ2_XXS (1.416 GiB) and bartowski's Q2_K (2.041 GiB). M (1.80 GiB) is an intermediate budget that matches no published file.
- "random" rows apply the allocator's upgrades in a random order (seeds 1–3). Like the greedy allocator, they stop when no upgrade fits the budget.
- "uniform ... body" rows put every tensor except the embedding and the DeltaNet gates at one type, including the sensitive attention and DeltaNet output projections. The gates stay at Q8_0 and the embedding at Q4_K, as in the floor (L: Q6_K, as in the ceiling). They protect no sensitive tensor.
- The `imatrix` column gives the calibration used: Unsloth's or bartowski's published matrix, or our own traces.
- `file_GiB` is the GGUF file size.
- Divergence columns are as in `sensitivity.csv`.
- `E_long_se_pct` and `E_chat_se_pct` are the binomial standard errors that `llama-perplexity` reports for each disagreement rate. They understate the true uncertainty, because neighbouring tokens are correlated.
- The row "greedy, own-trace imatrix" is the recipe released as magi-1.0. Its 1.7972 GiB is the size of the released file.
- The row "… full-attention k/v/output at Q6_K (= magi-beta-1)" was measured on a build with the released magi-beta-1's tensor types, importance matrix and file size (1.8275 GiB). It is not an allocator output.

**`candidates.csv`: every candidate of the search for a smaller global allocation.**
- Evidence: MEASURED; `J_pct` and the guard result are INFERRED.
- Backs Table 9 and Section 6.
- Each candidate was assembled from tensors quantized with Unsloth's importance matrix and measured on E-long and E-chat. `file_GiB` is the size of that file. The reference row is greedy M with the same matrix.
- Guard: J ≤ 16.36, which is 1.6 times the reference. It was fixed before any candidate was measured.
- Only candidate d was screened (built with the own-trace matrix). The screen columns and `gate` give its result; `chat_flags_of_48` counts flagged replies in the chat smoke test that was part of the gate.

**`instrument.csv`: the divergence instrument on published files.**
- Evidence: MEASURED.
- Backs Table 2.
- E-long and E-chat as above. E-code: 9,216 scored tokens of 46 LiveCodeBench problem statements dated before 2025-01-18. E-code is not part of J.
- `E_code_ppl_ratio_vs_BF16` is the perplexity ratio on E-code, which shows why perplexity was not used (Q3_K_M scores below 1).

## Benchmark items

`task_id` identifies the public item each task was built from. All prompts were zero-shot.

| Suite | Source | Selection | `task_id` |
|---|---|---|---|
| HumanEval+, MBPP+ | EvalPlus package | all 164 / 399 problems | EvalPlus task id |
| LiveCodeBench | `LiveCodeBench/code_generation_lite`, release_v6, test | the 150 newest by contest date (2025-01-18 to 2025-04-06) | `lcb:` + question id |
| ARC-Challenge | `allenai/ai2_arc` ARC-Challenge, test | first 150 by id (at most 5 choices) | `arc/` + id |
| MMLU | `cais/mmlu`, test | first 20 items of 10 subjects | `mmlu/<subject>/<index>` |
| HellaSwag | `Rowan/hellaswag`, validation | first 200 | `hellaswag/` + `ind` |
| WinoGrande | `allenai/winogrande` winogrande_xl, validation | first 200 by sentence | `winogrande/` + position |
| OpenBookQA | `allenai/openbookqa` main, test | first 200 by id | `obqa/` + id |
| SciQ | `allenai/sciq`, test | first 200 by question text | `sciq/` + position |
| TruthfulQA | `truthfulqa/truthful_qa` multiple_choice (mc1), validation | first 200 by question text (at most 5 choices) | `truthfulqa/` + position |
| GSM8K | `openai/gsm8k` main, test | first 150 | `gsm8k/` + index |
| ToolCall | in-house, generated from 5 tool templates | 100 | `toolcall/` + index |

The evaluation harness, prompt templates, answer-extraction rules and the ToolCall suite are not released yet.
