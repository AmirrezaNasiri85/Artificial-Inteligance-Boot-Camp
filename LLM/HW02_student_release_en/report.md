# HW02 Report — From Logits to Language

Full name: …

Student ID: …

Every answer must cite at least one of your own generated files, table rows, numerical values, or figures. Claims such as “top-p is always better,” “higher temperature is always more creative,” “second-best gives high-quality diversity,” or “greedy finds the best sentence” are not accepted without evidence.

## 1. Logits versus probabilities

Choose one token from `top_tokens.csv` and interpret its logit, probability, rank, and cumulative probability.

## 2. Why stable softmax is numerically stable

Report the naive and stable results from `softmax_stability.json`, then prove that subtracting a shared constant does not change the probabilities.

## 3. Temperature, entropy, and top-token probability

Use numerical values to compare `entropy_nats`, `top_token_probability`, and `support_size_90` across all four temperatures. Also examine whether rank order remains fixed.

## 4. Greedy versus second-best

Calculate the rank-two to rank-one probability ratio for distributions A and B. Why does rank two not represent a fixed uncertainty level?

## 5. Top-k versus top-p

Use two rows from `sampling_candidate_sets.csv` to demonstrate the difference in candidate-set rules and the difference between model and sampling probabilities.

## 6. Seed effects

Compare two runs with the same seed and one with a different seed using token IDs. State separately whether each policy is deterministic or stochastic.

## 7. Generation-trace analysis

Use `generation_trace.csv` to show that each step uses one forward pass, the final-position logits, and one new token. Explain context cropping.

## 8. Deliberately defective second-best decoding

Compare rank, model probability, cumulative log-probability, and repetition for the two paths. Does second-best occasionally produce a locally plausible token? Why is that insufficient?

## 9. Local optimum versus sequence optimum

Calculate the two path probabilities and log-probabilities in `toy_sequence_search.json`. Explain why exhaustive search is impractical for realistic vocabulary sizes and generation depths.

## 10. Decoding-metric comparison

Compare at least three policies by likelihood, rank, entropy, candidate count, distinct-1/2, and repetition. What limits these metrics as measures of text quality?

## 11. Adaptive-decoder design and evaluation

Report calibration values, threshold-selection method, held-out branch rates, and differences from fixed greedy and top-p. Why is this heuristic not optimal?

## 12. Experimental limitations

Explain the effects of the byte tokenizer, small repetitive model, six prompts, three seeds, absence of human evaluation, and difference between high probability and global correctness or coherence.
