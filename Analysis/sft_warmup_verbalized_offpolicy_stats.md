# Fixed Verbalized Off-Policy SFT Warm-Up Summary Stats

Final dataset: `/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Analysis/branching_eval/sft_warmup_off_policy/sft_warmup_final_sft2000_ckpt300_verbalized_offpolicy_p02_nobranch_fixed_complete_2000_20260623T131907Z`

## System Prompt And Final Answer Tag Check

| artifact field | rows |
|---|---:|
| jsonl_prompt_rows_with_system | 0 |
| jsonl_messages_rows_with_system | 0 |
| final_answer_rows_with_control_tags | 0 |

Result: final JSONL/parquet rows contain no system prompt rows and no post-`</think>` final answers containing `<think>`, `<steer>`, or `<exec>` control tags.

## Row Counts

| dataset | rows |
|---|---:|
| final generated cleaned | 2000 |
| original full filtered source | 2193 |
| original rows matched to final source indices | 2000 |

## Length Stats

| dataset | metric | mean | median | p10 | p90 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| generated | tokens | 2793.34 | 2332.00 | 1247.90 | 4928.30 | 592.00 | 12912.00 |
| generated | chars | 9934.70 | 8552.50 | 4593.60 | 17041.60 | 1655.00 | 36810.00 |
| generated | words | 1537.63 | 1311.50 | 679.00 | 2674.50 | 71.00 | 6624.00 |
| original matched | tokens | 3298.72 | 2679.50 | 1335.80 | 5700.30 | 511.00 | 17080.00 |
| original matched | chars | 11514.69 | 9748.00 | 4927.70 | 19723.10 | 1094.00 | 66430.00 |
| original matched | words | 1803.55 | 1527.00 | 722.90 | 3075.40 | 61.00 | 10375.00 |
| original full | tokens | 3386.37 | 2743.00 | 1340.20 | 5998.60 | 511.00 | 17080.00 |
| original full | chars | 11708.54 | 9906.00 | 4971.20 | 20093.60 | 1094.00 | 66430.00 |
| original full | words | 1839.97 | 1547.00 | 729.00 | 3148.20 | 61.00 | 10375.00 |

## Generated / Original Matched Length Ratios

| ratio | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| token_ratio | 0.94 | 0.87 | 0.56 | 1.37 | 0.23 | 5.46 |
| char_ratio | 0.96 | 0.88 | 0.57 | 1.39 | 0.21 | 4.57 |
| word_ratio | 0.95 | 0.87 | 0.56 | 1.40 | 0.21 | 5.36 |

## Domain / Source Distribution

| inferred domain | final rows | final % | original full rows | original full % |
|---|---:|---:|---:|---:|
| code | 1 | 0.1% | 3 | 0.1% |
| general/chat | 422 | 21.1% | 442 | 20.2% |
| general/reasoning | 1 | 0.1% | 1 | 0.0% |
| math | 40 | 2.0% | 44 | 2.0% |
| multilingual/general | 4 | 0.2% | 5 | 0.2% |
| python/code | 541 | 27.1% | 604 | 27.5% |
| science | 655 | 32.8% | 679 | 31.0% |
| synthetic/general | 336 | 16.8% | 415 | 18.9% |

### Source Distribution

| source | final rows | original full rows |
|---|---:|---:|
| `saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2` | 655 | 679 |
| `saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2` | 541 | 604 |
| `allenai/oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered` | 422 | 442 |
| `allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered` | 336 | 415 |
| `saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2` | 40 | 44 |
| `allenai/aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered` | 4 | 5 |
| `saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2` | 1 | 3 |
| `allenai/coconot-r1-format-domain-filtered-keyword-filtered-filter-datecutoff-chinese-filtered` | 1 | 1 |

## Wait / Alternatively Frequency

| dataset | wait total | rows with wait | wait / 1k tokens | alternatively total | rows with alternatively | alternatively / 1k tokens |
|---|---:|---:|---:|---:|---:|---:|
| generated | 6302 | 1046 (52.3%) | 1.128 | 3090 | 893 (44.6%) | 0.553 |
| original matched | 3439 | 579 (28.9%) | 0.521 | 1954 | 688 (34.4%) | 0.296 |
| original full | 3903 | 631 (28.8%) | 0.526 | 2261 | 751 (34.2%) | 0.304 |

## Generated Per-Domain Length And Discourse Markers

| domain | rows | mean tokens | median tokens | wait total | alternatively total |
|---|---:|---:|---:|---:|---:|
| code | 1 | 4790.0 | 4790.0 | 21 | 4 |
| general/chat | 422 | 2027.7 | 1824.5 | 476 | 278 |
| general/reasoning | 1 | 2053.0 | 2053.0 | 5 | 1 |
| math | 40 | 6076.0 | 5664.5 | 586 | 321 |
| multilingual/general | 4 | 2342.0 | 2037.5 | 1 | 1 |
| python/code | 541 | 2141.6 | 1900.0 | 2459 | 679 |
| science | 655 | 2828.3 | 2562.0 | 1653 | 1406 |
| synthetic/general | 336 | 4347.1 | 3866.0 | 1101 | 400 |

## Structure Stats

| block type | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| steer | 17.43 | 16.00 | 10.00 | 27.00 | 4.00 | 85.00 |
| exec | 17.43 | 16.00 | 10.00 | 27.00 | 4.00 | 85.00 |

Raw machine-readable stats: `/users/PAA0201/ollieproudman/work/DecomposedReasoning/Analysis/sft_warmup_verbalized_offpolicy_stats.json`
