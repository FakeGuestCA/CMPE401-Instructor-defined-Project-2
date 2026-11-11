# CMPE401 Instructor-defined Project 2

Group members: Fortune Olawale, Trevor Wu.

# 1. Project Overview
What the project investigates

# 2. Objectives
What I intended to learn

# 3. Datasets
FordA and Jena Climate

# 4. Model Architectures
Transformer and LSTM explanations

For the given LSTM:
1. Input: 120 hourly observations * 7 features each
2. LSTM layer: 32 units
3. Output: 1 output (predict temperature for next 12 hours)

# 5. Baseline Results
Results from the two original notebooks

Note that for the Transformer notebook, there is an inconsistency with the "GlobalAveragePooling1D" layer. The provided code produces a pooling output shape of "(None, 1)", while the model summary published on the official Keras shows an output shape of "(None, 500)". This might have limited the effectiveness of the attention and normalization layer, causing the training to stop early and achieve an accuracy of ~51% with a loss of ~0.693.

For the LSTM, in the training and validation loss graph, I noticed that the model started to overfit after 8 epochs, as the training loss continues to decrease while the validation loss starts to increase. This means that the model started to learn the training noise/details rather than the general patterns.

# 6. Improvement Experiments
For the improved experiments, I chose modifying the LSTM Forcasting.

For the first experiment, I increased the batch size from 256 to 512.
For the second experiment, I increased the learning rate from 0.001 to 0.002.
For the final experiment, I stacked the two LSTM layers.

All three of the notebooks with outputs are included. 

# 7. Benchmark Comparison
Tables and graps comparing results.
*Insert the photos of the graphs from the baseline + 3 experiments.

# 8. Discussion
What I learned and what worked best

Which model did you find easier to understand and why?

What improvement did you try, and what did you learn from it?
1. For the first experiment (increasing batch size), I noticed that the validation loss didn't decrease after epoch 3. After epoch 3, as more epochs are completed, the loss decreased from 0.1178 to 0.1064. However, the validation loss steadily increased from 0.1211 to 0.1255. This means the model started to overfit after epoch 3, meaning that it learned the training details rather than general patterns. Additionally, in the training and validation loss graph, both of the curves are much smoother and do not have as many jumps compared to the baseline graph, which makes sense as higher batch size means that the parameters are not updated as frequently.
2. In the second experiment (increasing learning rates), the validation loss starts off noticeable higher and it steadily decreases at a higher rate as the more epochs are completed than the baseline. Neither the loss numbers nor the training and validation loss graph showed any signs of the model overfitting like in experiment 1, because even though the learning rate has been doubled, it might still not be large enough to have an impact.
3. In the final experiment (stacking two LSTM layers), while it took the longest to train, it has the lowest training loss of 0.1010 as well as validation loss of 0.125, compared to the baseline as well as the two previous experiments. This means that the model is more accurate and generalize better after stacking two LSTM layers together. Looking at the training and validation loss graph, it is noticeable not as jumpy as the baseline, which might have been also caused by the two LSTM layers.

# 9. Conclusion
Overall findings
