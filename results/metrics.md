# Evaluation results

All rows are scored on the same 75 films (labelled, and rated by every method shown). Labels extracted from the reviews with `qwen/qwen3.8-27b`.

## Overall (4 categories pooled)

| Method | Model | Films | Exact | 95% CI | Within one level | Weighted κ | Evidence found in subtitles | Mean tool calls | List-price cost |
|---|---|---|---|---|---|---|---|---|---|
| Full transcript (no retrieval) | `gpt-5-mini` | 75 | **52%** | 46%–58% | 92% | 0.46 | 95% of 745 | 0.0 | $0.43 |
| RAG, fixed query per category | `gpt-5-mini` | 75 | **48%** | 42%–54% | 92% | 0.41 | 98% of 809 | 0.0 | $0.30 |
| Agentic RAG (search tools) | `gpt-5-mini` | 75 | **50%** | 44%–56% | 92% | 0.44 | 98% of 785 | 0.4 | $0.34 |
| Majority label per category | — | — | 47% | — | 86% | 0.17 | — | — | — |

## Exact agreement by category

| Category | Majority baseline | Full transcript (no retrieval) | RAG, fixed query per category | Agentic RAG (search tools) |
|---|---|---|---|---|
| Violence & Scariness | 49% (strong) | 52% (κ 0.43) | 36% (κ 0.26) | 43% (κ 0.28) |
| Sex, Romance & Nudity | 55% (moderate) | 56% (κ 0.37) | 48% (κ 0.22) | 51% (κ 0.33) |
| Language | 47% (strong) | 60% (κ 0.49) | 68% (κ 0.65) | 71% (κ 0.64) |
| Drinking, Drugs & Smoking | 39% (moderate) | 40% (κ 0.33) | 39% (κ 0.26) | 36% (κ 0.25) |

## Review label distribution

| Category | none | mild | moderate | strong | from empty field |
|---|---|---|---|---|---|
| Violence & Scariness | 2 | 12 | 24 | 37 | 2 |
| Sex, Romance & Nudity | 5 | 17 | 41 | 12 | 6 |
| Language | 3 | 13 | 24 | 35 | 3 |
| Drinking, Drugs & Smoking | 7 | 18 | 29 | 21 | 7 |

## Label reliability

A second, independent label pass with `openai/gpt-oss-20b` agreed with the main labels on 74% of 300 film-category pairs (within one level: 99%, weighted κ 0.82).

## Mean signed error (prediction minus label, in levels)

| Category | Full transcript (no retrieval) | RAG, fixed query per category | Agentic RAG (search tools) |
|---|---|---|---|
| Violence & Scariness | -0.12 | -0.33 | -0.29 |
| Sex, Romance & Nudity | -0.16 | -0.24 | -0.28 |
| Language | +0.39 | +0.13 | +0.17 |
| Drinking, Drugs & Smoking | -0.19 | -0.21 | -0.21 |

Negative means the pipeline rates milder than the review.
