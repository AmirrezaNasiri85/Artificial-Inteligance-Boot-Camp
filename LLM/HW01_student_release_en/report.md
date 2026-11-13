# HW01 Report

Full name:

Student ID:

Every answer must cite your own evidence, such as a filename, row number, measured value, or figure.

1. How many supervised predictions did your training forward pass produce? Calculate the value from the shape and `supervised_positions`.
2. With the prefix fixed and future tokens changed, what was the maximum logit difference with and without the causal mask? Why does this test reveal leakage more directly than comparing the loss of a randomly initialized model?
3. What were the initial, final, and validation losses in the short training run? What limits this comparison?
4. How many forward passes were required to generate the new tokens? Analyze one trace row to explain why generation uses the final-position logits.
5. After injecting the incorrect token, what were the maximum and final total-variation distances? Why can every later conditional distribution change?
6. Use code or measured evidence to explain why `model.eval()` and `torch.inference_mode()` do not perform the same operation.
7. Compare dropout, gradients, optimizer use, and memory requirements in the two regimes.
8. Why is MiniGPT decoder-only in both regimes, and why is the idea that training uses an encoder that inference removes incorrect?
9. Record exact versions, seeds, CPU time, and every permitted configuration change.
10. Disclose all sources of assistance or collaboration.
