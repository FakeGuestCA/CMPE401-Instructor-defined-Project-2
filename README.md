# CMPE 401 Instructor-defined Project 2
# Understand, Organize Time Series Modelling Benchmarks for Classification and Forecasting

Group members: Fortune Olawale, Trevor Wu.

# 1. Project Overview
This project explores two time-series modelling tasks using official Keras examples:
1. Time-series classification: A Transformer model trained on FordA dataset.
2. Time-series forecasting: A LSTM (Long Short-Term Memory) model trained on Jena Climate dataset.

Time-series data is commonly used in engineering and real-world applications, including weather monitoring, industrial sensing, health monitoring, traffic analysis, and activity recognition.

The project focuses on understanding how these two neural network architectures work, reproducing the provided implementations, and investigating how modifications to model configurations and architecture affect training behaviour and predictive performance.

**The LSTM forecasting model was selected for the improvement experiments. Details below.**

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
Two datasets are used in this project.

## 3.1 FordA
FordA is used for the Transformer-based time-series classification task.

The dataset contains time-series measurements represented as sequences of 500 observations with one feature per time step. The objectives is to classify each sequence into one of two classes.

## 3.2 Jena Climate
The Jena Climate dataset is used for the LSTM-based forecasting task.

The dataset was recorded at the weather station of the Max Planck Institute for Biogeochemistry in Jena, Germany, between January 2009 and December 2016. It contains weather measurements such as temperature, atomspheric pressure, humidity, and wind speed.

The original dataset contains 14 features recorded every 10 minutes. For this project, seven selected features are used, and the data is sampled hourly.

The LSTM has:
- Input sequence: 120 time steps, with 7 features per time step.
- Historical window: Approximately 5 days of weather observations.
- Prediction target: Temperature 12 hours after the input window.

The dataset is normalized before training to help the model earn from features with different numerical ranges.

# 4. Model Architectures
## 4.1 Transformer
The Transformer uses self-attention mechanisms to identify relationships between different positions in a sequence. Unlike recurrent networks (such as LSTM), it can process sequence positions in parallel during training.

The provded implementation uses Transformer encoder blocks containing multi-head attention and feed-forward layers, followed by global average pooling and dense classification layers.

The model is trained to classify FordA sequences into two classes using sparse categorical cross-entropy as its loss function.

## 4.2 LSTM
The LSTM is a type of recurrent neural network designed to learn temporal relationships in sequential data. It uses gates to control which information is retained, updated, or discarded over time.

The original Keras implementation consists of:
1. Input: 120 time steps, each containing 7 weather features.
2. LSTM layer: 32 units that process the input sequence and extract temporal information.
3. Dense layer: One output neuron that produces the predicted temperature.

The model uses the Adam optimizeer with a learning rate of 0.001 and mean squared error (MSE) as its loss function.

## 4.3 Transformer vs. LSTM
| Feature | Transformer | LSTM |
| ------- | ----------- | ---- |
|Data processing|Can process sequence positions in parallel using self-attention.|Processes sequences recurrently, one time step at a time.|
|Key mechanism|Self-attention using queries, keys, and values.|Input, forget, and output gates.|
|Training speed|Parallel processing can improve GPU utilization.|Sequential dependencies can limit parallel processing.|
|Long-range context|Directly relates distant positions through attention.|Retains information through hidden and cell states, although very long dependencies can be challenging.|
|Computational characteristics|Self-attention has quadratic scaling with sequence length.|Recurrent computation scales approximately linearly with sequence length, but is sequential.|

Neither architecture is universally faster or more efficient. Their performance depends on the sequence length, model size, hardware, and implementation.

# 5. Baseline Results
The Colab notebooks were used to establish baseline results before performing any modifications.
The baseline CSV file is available at **results/baseline_transformer_and_lstm.csv**.

| Measurement | Transformer | LSTM |
| ----------- | ----------- | ---- |
|Dataset|FordA|Jena Climate|
|Maximum epochs|150|10|
|Actual epochs competed|11|10|
|Batch size|64|256|
|Learning rate|0.0001|0.001|
|Recorded training loss|0.6928|0.1058|
|Recorded validation loss|0.6927|0.1295|
|Test performance (accuracy)|0.5159|N/A|
|Sparse categorical accuracy (SCA)|0.5122|N/A|
|Validation SCA|0.5146|N/A|
|Validation MAE|N/A|0.2853|
|Validation MSE|N/A|0.13149|
|Training time|9 minutes|10 minutes|

**Transformer baseline observation:**
The Transformer training stopped early after 11 epochs because the validation loss did not improve. Its test accuracy was approximately 51.59%, with a loss close to 0.693.

During investigation, an inconsistency was identified in the `GlobalAveragePooling1D` layer. The provided source code specifies `data_format="channels_last"`, producing an output shape of of `(None, 1)` for an input shape of `(None, 500, 1)`. However, the model summary published on the official Keras example shows an output shape of `(None, 500)`.

This discrepancy was documented and reported to the instructor (via email). The instructor comfirmed that retaining the original implementation and its results as the baseline was acceptable. The results are therefore retained without making architectural corrections.

**LSTM baseline observation:**
The training loss generally decreased throughout the 10 epochs, while the validation loss fluctuated and eventually increased after reaching a lower value earlier in training.

This suggests that the model may have started overfitting, as it continued improving its fit to the training data without achieving corresponding improvements on the validation data.

However, the baseline was only run once. Therefore, additional runs would be necessary to determine how consistently this behaviour occurs.

# 6. Improvement Experiments
The LSTM forecasting model was selected for the improvement task. Three experiments were performed, with one major variable changed at a time while keeping the remaining configuration unchanged.

## 6.1 Experiment 1: Increasing Batch Size
**Modification:**
- Original batch size: 256
- Modified batch size: 512
- Learning rate: 0.001
- Architecture: One LSTM layer with 32 units.

**Observations:**
The model stopped early after 8 out of 10 maximum epochs. The training log shows that the validation loss reached its lowest value of approximately 0.1211 at epoch 3. It then increased over the following epochs, while the training loss continued to decrease.

The early stopping callbacked used a patience value of 5. Since the validation loss did not improve for 5 consecutive epochs after epoch 3, training stopped after epoch 8. (Even though early stopping is outside of the scope of this project as mentioned by the instructor, I have decided to look into it)

The training and validation curves also appeared smoother compared with the baseline. This may be explained by the larger batch size. Increasing the batch size reduces the number of batches processed per epoch, meaning fewer weight updates occur during training. Each update is also calculated using more training samples.

The final recorded validation loss was 0.1255, even though the best validation loss was 0.1211 at epoch 3.

Training time decreased from approximately 10 minutes to 6 minutes. This suggests that the larger batch size reduced the overall training time in this particular experiment, although the experiment also completed fewer epochs.

## 6.2 Experiment 2: Increasing Learning Rate
**Modification:**
- Batch size: 256
- Original learning rate: 0.001
- Modified learning rate: 0.002
- Architecture: One LSTM layer with 32 units.

**Observations**
The training loss decreased to 0.1023, compared with 0.1058 in the baseline.

However, the final validation loss increased from the baseline of 0.1295 to 0.1349. This indicates that the lower training loss didn not translate into better validation performance.

The validation loss started relatively high at 0.1661 and generally decreased throughout training, reaching 0.1349 by epoch 10. There was a small increase at epoch 4, but the overall trend was downwards.

The higher learning rate did not cause obvious training instability in this experiment. However, it also did not improve the model's validation performance.

This demonstrates that increasing the learning rate does not necessarily improve model performance. Although the model achieved a lower training loss, its performance on validation data was worse than the baseline.

## 6.3 Experiment 3: Stacking Two LSTM Layers
**Modification:**
- Batch size: 256
- Learning rate: 0.001
- Original architecture: One LSTM layer with 32 units.
- Modified architecture: Two stacked LSTM layers, each with 32 units.

**Observations:**
The first LSTM layer was configured with `return-sequence=True` so that it would return its output at every time step. This sequence was then passed to the second LSTM layer, which produced the final output used by the Dense layer.

The modified model achieved a training loss of 0.1010 and a validation loss of 0.1250.

Compared with the baseline, both the training and validation losses decreased. The validation loss also generally decreased throughout the 10 epochs, with only a small increase at epoch 9.

The stacked model had 13,473 parameters, compared with 5,153 parameters in the original model. This is approximately 2.61 times as many parameters.

Training time also increased slightly from approximately 10 minutes to 11 minutes.

The results suggest that adding a second LSTM layer improved the model's validation performance in this particular run, although it required more parameters and slightly more training time.

# 7. Benchmark Comparison
The following table summarizes the results of the baseline and all three experiments.
|Configuration|Epochs|Training loss|Final validation loss|Best validation loss|Training time|
|---|---|---|---|---|---|
|Baseline|10|0.1058|0.1295|0.1276|10 min|
|Batch size 512|8|0.1064|0.1255|0.1211(epoch 3)|6 min|
|Learning rate 0.002|10|0.1023|0.1349|0.1349|9 min|
|Two LSTM layers|10|0.1010|0.1250|0.1250|11 min|

## 7.1 Training and Validation Loss Graphs
**Baseline**
![Baseline](https://github.com/FakeGuestCA/CMPE401-Instructor-defined-Project-2/blob/main/results/graphs/lstm_baseline_training_and_validation_graph.png?raw=true)
**Experiment 1: Batch Size 512**
![Batch size 512](https://github.com/FakeGuestCA/CMPE401-Instructor-defined-Project-2/blob/main/results/graphs/lstm_modified1_training_and_validation_graph.png?raw=true)
**Experiment 2: Learning Rate 0.002**
![Learning rate 0.002](https://github.com/FakeGuestCA/CMPE401-Instructor-defined-Project-2/blob/main/results/graphs/lstm_modified2_training_and_validation_graph.png?raw=true)
**Experiment 3: Two Stacked LSTM Layers**
![Two stacked LSTM layers](https://github.com/FakeGuestCA/CMPE401-Instructor-defined-Project-2/blob/main/results/graphs/lstm_modified3_training_and_validation_graph.png?raw=true)

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
