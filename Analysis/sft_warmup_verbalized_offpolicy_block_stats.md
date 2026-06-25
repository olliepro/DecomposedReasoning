# Fixed Verbalized Off-Policy SFT Warm-Up Block Length Stats

Dataset: `/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Analysis/branching_eval/sft_warmup_off_policy/sft_warmup_final_sft2000_ckpt300_verbalized_offpolicy_p02_nobranch_fixed_complete_2000_20260623T131907Z`

Lengths below measure block contents only; `<steer>`, `</steer>`, `<exec>`, and `</exec>` tags are excluded.
Exact token counts use tokenizer: `/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/qwen35_branching_dapo/served_models/qwen35_4b_branch_gs50_structured_baseline_lr2e6_execT1_5611097_global_step_300_hf`

## Parse Check

- rows: `2000`
- parsed blocks: `69704`
- rows where parsed block count did not match stored steer+exec count: `0`

## Average Block Length By Type

| block | blocks | mean tokens | median tokens | p10 tokens | p90 tokens | mean words | median words | mean chars | median chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| steer | 34852 | 5.27 | 5.00 | 3.00 | 8.00 | 3.93 | 4.00 | 28.13 | 27.00 |
| exec | 34852 | 106.98 | 82.00 | 39.00 | 193.00 | 67.25 | 56.00 | 393.57 | 333.00 |

## Distribution Details

| block | metric | mean | median | p25 | p75 | p90 | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steer | tokens | 5.27 | 5.00 | 4.00 | 6.00 | 8.00 | 9.00 | 2.00 | 94.00 |
| steer | words | 3.93 | 4.00 | 3.00 | 4.00 | 5.00 | 6.00 | 1.00 | 55.00 |
| steer | chars | 28.13 | 27.00 | 23.00 | 32.00 | 37.00 | 41.00 | 8.00 | 389.00 |
| exec | tokens | 106.98 | 82.00 | 56.00 | 123.00 | 193.00 | 276.00 | 2.00 | 516.00 |
| exec | words | 67.25 | 56.00 | 39.00 | 81.00 | 118.00 | 153.00 | 1.00 | 436.00 |
| exec | chars | 393.57 | 333.00 | 225.00 | 478.25 | 690.00 | 886.00 | 8.00 | 2825.00 |

## Synthetic Off-Policy Steer Blocks

| subtype | blocks | mean tokens | median tokens | mean words | median words |
|---|---:|---:|---:|---:|---:|
| synthetic_enumerate_steer | 651 | 14.00 | 14.00 | 9.00 | 9.00 |
| synthetic_continue_steer | 687 | 4.00 | 4.00 | 3.00 | 3.00 |
| normal_steer | 33514 | 5.13 | 5.00 | 3.85 | 4.00 |
| normal_exec | 34852 | 106.98 | 82.00 | 67.25 | 56.00 |

## Blocks Per Completion

| block | mean per row | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| steer | 17.43 | 16.00 | 10.00 | 27.00 | 4.00 | 85.00 |
| exec | 17.43 | 16.00 | 10.00 | 27.00 | 4.00 | 85.00 |

Raw machine-readable stats: `/users/PAA0201/ollieproudman/work/DecomposedReasoning/Analysis/sft_warmup_verbalized_offpolicy_block_stats.json`
