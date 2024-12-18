# Comedy Data Analysis with Machine Learning

This project analyzes data from comedy performances to predict the "HumanScore" of jokes using machine learning models. The dataset includes features such as intensity, pitch, and performance metadata. By engineering new features and applying various ML algorithms, this project explores the relationships between joke quality, audience reactions, and performance biases.

---

## Objectives

1. **Feature Engineering**:
   - Derive meaningful features like intensity range, total intensity, and weighted HumanScores.
   - Analyze biases in performances and audiences.

2. **Visualization**:
   - Box plots, scatter plots, and bar charts to highlight performance metrics and joke scores.

3. **Model Implementation**:
   - Train machine learning models including Decision Trees, k-Nearest Neighbors, and Support Vector Machines.
   - Use stratified k-fold cross-validation to evaluate model performance.

4. **Performance Comparison**:
   - Identify the most effective model for predicting HumanScores.
   - Compare the impact of features like MFCC (Mel-Frequency Cepstral Coefficients) on accuracy.

---

## Technologies

- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation and analysis.
  - `matplotlib`: Visualization.
  - `sklearn`: Machine learning models and metrics.

---

## Data Processing

1. **Feature Engineering**:
   - Created range and total features for intensity, pitch, and MFCCs.
   - Weighted HumanScores based on joke and performance averages to reduce biases.

2. **Visualization**:
   - Visualized the relationships between features and HumanScore using box plots and scatter plots.

3. **Model Training**:
   - Models were trained using various feature combinations, including the top features identified by Random Forest and Permutation Importance.

---

## Results

1. **Top Features**:
   - `intensity_mean`, `intensity_std`, `2_mean`, `length`, and `intensity_total` were identified as the most significant predictors.

2. **Model Performance**:
   - **Support Vector Machines (SVM)** achieved the highest accuracy (~85%), followed by k-NN and Decision Trees.
   - Models trained with the top 10 features outperformed those with all features.

3. **Impact of MFCC**:
   - Models performed well even without MFCC data, suggesting that most MFCC data might not be essential.

