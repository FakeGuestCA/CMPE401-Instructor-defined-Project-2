# CMPE 401 Instructor-defined Project 2
# Understand, Organize Time Series Modelling Benchmarks for Classification and Forecasting

Group members: Fortune Olawale, Trevor Wu.

# 1. Project Overview
This project explores two time-series modelling tasks using official Keras examples:
1. Time-series classification: A Transformer model trained on FordA dataset.
2. Time-series forecasting: A LSTM (Long Short-Term Memory) model trained on Jena Climate dataset.

Time-series data is commonly used in engineering and real-world applications, including weather monitoring, industrial sensing, health monitoring, traffic analysis, and activity recognition.

The project focuses on understanding how these two neural network architectures work, reproducing the provided implementations, and investigating how modifications to model configurations and architecture affect training behaviour and predictive performance.

The LSTM forecasting model was selected for the improvement experiments.

# 2. Objectives
The main objectives of this project are to:
1. Understand the differences between time-series classification and time-series forecasting.
2. Understand the differences between a Transformer-based model and an LSTM-based model.
3. Reproduce and run the provided Keras examples.
4. Understand the dataset, model input and output, training processes, and evaluation metrics.
5. Investigate how changes to training configurations affect model peroformance.
6. Modify an existing model architecture and evaluate the effects of the modification.
7. Organize the notebooks, results, and observations into a documented GitHub repository.

# 3. Datasets
For datasets, FordA is used for the Transformer classification, and Jena Climate is used for the LSTM forecasting.

# 4. Model Architectures
Major differences between Transformer and LSTM
| Feature | LSTM | Transformer |
| ------- | ----------- | ---- |
|Data processing|Sequential: Data is processed one after another|Parallel: All data is processed simultaneously|
|Key mechanism|Uses input, forget, and output gates to either retain or discard information.|Direct relationships between all tokens are computed at once.|
|Training speed|Slower since data is processed step-by-step.|Faster since parallelism allows better GPU utilization.|
|Long-range context|Struggles over very long distance as information can dilute or fade over extended sequences.|Excels as self-attention connects any two points in the sequence regardless of distance.|
|Resource demand|Lower: Lightweight and more memory-efficient.|Higher: Quadratic memory/comput scaling relative to sequence length.|

For the given Transformer:


For the given LSTM:
1. Input: 120 hourly observations * 7 features each
2. LSTM layer: 32 units
3. Output: 1 output (predict temperature for next 12 hours)

# 5. Baseline Results
The CSV files for the baseline (and experiments) results can be found in the results folder. I will list them below as well.
| Measurement | Transformer* | LSTM |
| ----------- | ----------- | ---- |
|Dataset|FordA|Jena Climate|
|Training epochs|150|10|
|Batch size|64|256|
|Learning rate|0.0001|0.001|
|Final training loss|0.6928|0.1058|
|Final validation loss|0.6927|0.1295|
|Test performance (accuracy)|0.5159|-|
|Sparse categorical accuracy (SCA)|0.5122|-|
|Validation SCA|0.5146|-|
|Validation MAE|-|0.2853|
|Validation MSE|-|0.13149|
|Training time|9 minutes|10 minutes|

*Note that for the Transformer notebook, there is an inconsistency with the "GlobalAveragePooling1D" layer. The provided code produces a pooling output shape of "(None, 1)", while the model summary published on the official Keras shows an output shape of "(None, 500)". This might have limited the effectiveness of the attention and normalization layer, causing the training to stop early (usually from 10 epochs to 50 epochs) and achieve an accuracy of ~51% with a loss of ~0.693.

For the LSTM, in the training and validation loss graph, I noticed that the model started to overfit after 8 epochs, as the training loss continues to decrease while the validation loss starts to increase. This means that the model started to learn the training noise/details rather than the general patterns. However, this can be an error, as the baseline notebooks are only ran once and there are only 10 epochs (small compared to Transformer's 150 epochs).

# 6. Improvement Experiments
For the improved experiments, I chose modifying the LSTM Forcasting.

For the first experiment, I increased the batch size from 256 to 512.
For the second experiment, I increased the learning rate from 0.001 to 0.002.
For the final experiment, I stacked the two LSTM layers.

All three of the notebooks with outputs are included. They are in notebooks/modifed with output

# 7. Benchmark Comparison
Tables and graps comparing results.
*Insert the photos of the graphs from the baseline + 3 experiments.

# 8. Discussion
What I learned and what worked best

Which model did you find easier to understand and why?
Personally I find the LSTM model easier to understand since as of writing this (Nov 11, 2026) we have not gone over Transformers in the lectures yet. I had to read into additional papers to better understand the Transformer model.

What improvement did you try, and what did you learn from it?
1. For the first experiment (increasing batch size), I noticed that the validation loss didn't decrease after epoch 3. After epoch 3, as more epochs are completed, the loss decreased from 0.1178 to 0.1064. However, the validation loss steadily increased from 0.1211 to 0.1255. This means the model started to overfit after epoch 3, meaning that it learned the training details rather than general patterns. Additionally, in the training and validation loss graph, both of the curves are much smoother and do not have as many jumps compared to the baseline graph, which makes sense as higher batch size means that the parameters are not updated as frequently.
2. In the second experiment (increasing learning rates), the validation loss starts off noticeable higher and it steadily decreases at a higher rate as the more epochs are completed than the baseline. Neither the loss numbers nor the training and validation loss graph showed any signs of the model overfitting like in experiment 1, because even though the learning rate has been doubled, it might still not be large enough to have an impact.
3. In the final experiment (stacking two LSTM layers), while it took the longest to train, it has the lowest training loss of 0.1010 as well as validation loss of 0.125, compared to the baseline as well as the two previous experiments. This means that the model is more accurate and generalize better after stacking two LSTM layers together. Looking at the training and validation loss graph, it is noticeable not as jumpy as the baseline, which might have been also caused by the two LSTM layers.

# 9. Conclusion
Overall findings
