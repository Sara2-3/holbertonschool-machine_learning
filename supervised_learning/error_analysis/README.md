# My ReadME

# My ReadME


Analysis of Each Scenario:
1. High Bias, High Variance

Both underfitting AND overfitting (unusual situation)
Model architecture is fundamentally flawed
Answer: B (Try a different architecture)

2. High Bias, Low Variance

Underfitting - model is too simple
Training error is high, but generalizes consistently
Need to increase model capacity
Answer: B,D (Try different architecture, Build deeper network)

3. Low Bias, High Variance

Overfitting - fits training data well but doesn't generalize
Large gap between training and validation error
Answer: C,E (Get more data, Use regularization)

4. Low Bias, Low Variance

Ideal scenario! Model fits well and generalizes well
Answer: F (Nothing - it's already performing well!)

File Content:
Create 5-error_handling:
B
B,D
C,E
F

Looking at the confusion matrices, let me analyze the performance:
Analysis:
Training Set:

Accuracy = (54 + 52 + 41) / 180 = 81.67%
Error = 18.33%

Validation Set:

Accuracy = (53 + 51 + 40) / 180 = 80%
Error = 20%

Human Performance:

Error < 14% pra te jete me mire se njerezit, pra ta nxjerre me larte

Key Observations:

Training error (18.33%) is higher than human error (14%)

This means the model cannot even fit the training data well enough to match human performance
This indicates HIGH BIAS (underfitting)


Gap between training and validation error:

20% - 18.33% = 1.67% (very small gap)
Model performs similarly on both sets
This indicates LOW VARIANCE



Conclusion:
The model has High Bias - it's underfitting and not even reaching human-level performance on the training set.