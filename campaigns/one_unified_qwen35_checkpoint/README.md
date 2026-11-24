# Qwen3.5 MAI Profile scoring in AgentX

## Run the synthetic test first

Submit `job.smoke.yml` from the candidate checkout to `NewsFeedL2_AML`. It uses the original `ranking` Singularity cluster and managed identity. It scores four synthetic predictions without model weights, datastore inputs, or dependency installation. Use a single run to check AML logs and outputs, or two AgentX rounds to check the evaluator gate.

| Round | `inputs.checkpoint` | Expected `smoke_exact_match` |
| --- | --- | --- |
| 1 | `checkpoint-3000` | 0.50 |
| 2 | `checkpoint-2000` | 0.75 |

Set the Test9 evaluator Invocation to `{"metric":"smoke_exact_match","direction":"maximize","threshold":0.75}`. The first round should fail the threshold and the second should pass it. Each AML run prints one `AML_RESULT_JSON=` line and writes `smoke_score.json` under the `metrics` output. Both carry `synthetic: true`.

## Switch to the real job

After the cloud smoke test passes, point the same Campaign graph at `job.yml` and set the evaluator Invocation to `{"metric":"mai_profile_exact_match","direction":"maximize"}`. Use round one as the baseline and compare round two with it. The threshold of 0.75 applies only to the synthetic test.

The real job reuses the `project.yaml` and `agent_pkg` mounted by `red_balloon_58226mf611`. It stages a private runtime directory and rewrites paths from the old mount. Both rounds evaluate the same `mai_profile_val.jsonl` from the current read-only mount. The held-out test file stays untouched. The job merges the selected LoRA adapter and runs the existing MAI Profile scoring pipeline. A missing, stale or invalid result fails the job. A valid run prints one `AML_RESULT_JSON=` line and writes `mai_profile_eval.json` under the `metrics` output.

Wuchen runs the smoke and real jobs with his own account. On the first real run, confirm the AML logs show the selected adapter was loaded, the MAI Profile validation split was scored, and the final log contains `mai_profile_exact_match` for 1,000 examples.

## Campaign graph

Use AgentX Graph Eval with two iterations, `brainstorm` as the proposal manager and `evaluate` as the evaluator. Point each task at the repository branch containing the two job files.

- `brainstorm` uses `generic-brainstorm`. Propose `checkpoint-3000` first, then `checkpoint-2000`.
- `develop` uses `generic-code-developer` and depends on `brainstorm`. Edit only the selected job YAML. Set the proposed checkpoint and a distinct round display name.
- `train` uses `aml-agent` and depends on `develop`. Submit the selected YAML from the candidate workspace to `NewsFeedL2_AML` without input overrides. Monitor the same AML job through any resume. Return the job name, terminal status, and final `AML_RESULT_JSON=` line verbatim.
- `evaluate` uses `aml-metric-evaluator` and depends on `train`. Configure the metric for the selected job as shown above. Treat missing evidence or a failed AML job as inconclusive.

## Final goal for the real job

Compare checkpoints checkpoint-3000 and checkpoint-2000 for Qwen3.5-4B on the same MAI Profile validation split. Maximize mai_profile_exact_match across two Azure ML evaluation rounds. Report the measured score and AML job name for each checkpoint and identify the best-scoring checkpoint. Report a tie if the scores are equal.
