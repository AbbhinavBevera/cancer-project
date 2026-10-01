# Project Explainer & Interview Guide

This guide breaks down the project in plain English so you can explain every aspect of it in an interview, review, or presentation.

---

## 1. What is this project in simple words?

When a doctor examines a tumour, knowing whether a tissue sample is cancerous and understanding what molecules drive the tumour is essential.
- **Transcriptomics** measures how active thousands of genes are inside cells (gene expression).
- **The problem:** We have thousands of genes (e.g., 1,500 to 20,000 features) but only a few hundred patients (samples). In machine learning, this is called the **"curse of dimensionality"** or the $p \gg n$ problem. If you are not careful, a model can easily memorize noise rather than learning real biology.
- **The solution:** This project builds an honest, leak-free machine learning pipeline that:
  1. Predicts whether a sample is tumour or normal.
  2. Discovers **biomarkers**—the specific genes most consistently responsible for that diagnosis.
  3. Uses **stability selection** (repeatedly fitting across resamples) to distinguish real biological signals from statistical flukes.

---

## 2. Walkthrough of the Pipeline

```
Raw Counts / Expression Matrix
             │
             ▼
[Step 1: Unsupervised Preprocessing]
- Drop genes with >10% missing values
- Median imputation for remaining gaps
- Filter out near-zero expression noise
- Log2 / CPM transformation (no labels used!)
             │
             ▼
[Step 2: Hold-Out Test Split]
- 20% held out once at the start (stratified)
- Saved to results/splits.json
- NEVER touched until the very final test evaluation
             │
             ▼
[Step 3: Nested Feature Selection & Cross-Validation]
- 5-Fold Stratified Cross-Validation on training data only
- Inside EVERY fold:
    * Standardize features (StandardScaler)
    * Select top K features via ANOVA F-test (SelectKBest)
    * Fit classifier (L1 Logistic Regression, Linear SVM, Random Forest)
- Ensures ZERO test information leaks into feature selection
             │
             ▼
[Step 4: Stability Selection Biomarker Ranking]
- Repeated Stratified K-Fold (50 resamples)
- Fit L1-penalized models and record non-zero gene coefficients
- Frequency score: What fraction of resamples picked this gene?
- Genes selected in >95% of resamples are robust candidates
             │
             ▼
[Step 5: Statistical Differential Expression]
- Classical Welch's t-test + Benjamini-Hochberg FDR correction
- Validates ML findings against traditional bioinformatics methods
```

---

## 3. 15 Questions You Might Be Asked (and How to Answer Them)

### Q1: What is "data leakage" and how did you prevent it?
> **Answer:** Data leakage happens when information from outside the training dataset is used to create or tune the model. In gene expression, selecting genes across all samples *before* splitting into cross-validation folds leaks information about the validation set, creating artificially high scores that fail in the real world. We prevented this by putting `SelectKBest` directly inside a scikit-learn `Pipeline`. The feature selector is re-fitted from scratch on every training fold and never sees the validation fold.

### Q2: Why did you prioritize Recall over Precision?
> **Answer:** In cancer screening and diagnosis, missing a true tumour (a False Negative) could delay lifesaving treatment and be fatal. A false alarm (False Positive) leads to follow-up testing. Therefore, we prioritize high Recall (identifying 100% of tumours) while maintaining acceptable precision.

### Q3: Why compare Logistic Regression, Linear SVM, and Random Forest?
> **Answer:** 
> - **L1-penalized Logistic Regression (Lasso):** Naturally drives non-important feature weights to exact zero, creating sparse, interpretable gene signatures.
> - **Linear SVM:** Effective at finding a maximum-margin hyperplane in high-dimensional linear feature spaces.
> - **Random Forest:** An ensemble of decision trees that captures non-linear gene-gene interactions and is robust to outliers.

### Q4: What is "stability selection"?
> **Answer:** With thousands of genes, a single run of a model might pick gene A today and gene B tomorrow just due to random noise. Stability selection fits models over many different resamples (e.g. 50 folds) and counts how frequently each gene gets chosen. Genes chosen 95–100% of the time are genuine, stable biomarker candidates.

### Q5: What is the synthetic dataset used for?
> **Answer:** The synthetic dataset (`data/synthetic/`) uses negative-binomial distributions to simulate real RNA-seq properties (overdispersion, varying sequencing depths, batch effects) with 40 known ground truth biomarkers. It provides a benchmark to verify that the pipeline correctly recovers true biomarkers (achieving 80% precision) before running on patient data.

### Q6: How do you know the model isn't just memorizing noise?
> **Answer:** We performed a **label-permutation sanity check**. When we randomly shuffled the diagnosis labels, the model's cross-validation ROC-AUC collapsed from 0.98 down to ~0.50 (random guessing). If a pipeline still gets a high score with scrambled labels, it proves there is a leak. Ours correctly dropped to random chance.

### Q7: Why did you use log2 transformation?
> **Answer:** RNA sequencing read counts follow a heavy right-skewed power-law distribution (a few genes have millions of reads; most have very few). The $\log_2(\text{count} + 1)$ transformation stabilizes variance, prevents hyper-expressed genes from dominating distance metrics, and matches biological fold-change calculations.

### Q8: What does Benjamini-Hochberg FDR correction do in Step 5?
> **Answer:** When testing 1,500 genes individually at $p < 0.05$, you would expect $1500 \times 0.05 = 75$ false positives purely by chance (the multiple testing problem). Benjamini-Hochberg adjusts the p-values to control the False Discovery Rate (FDR), ensuring that no more than 5% of reported discoveries are false positives.

### Q9: What are the main limitations of this study?
> **Answer:** 
> 1. It is currently demonstrated on synthetic benchmarks and needs external clinical validation across diverse patient cohorts.
> 2. Microarray or RNA-seq batch differences between hospitals require batch-correction techniques.
> 3. Discovered genes are computational *candidates* that require wet-lab biological validation (RT-qPCR or functional assays).

### Q10: How do you handle missing values in gene expression?
> **Answer:** Genes missing in more than 10% of samples are dropped entirely. For the remaining mild missingness, we impute using the median of that gene across available samples. Crucially, imputation is done in an unsupervised manner without looking at patient diagnosis labels.

### Q11: What is a ROC curve and ROC-AUC?
> **Answer:** A Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Recall) vs False Positive Rate at every possible classification threshold. The Area Under the Curve (AUC) measures aggregate classification performance across all thresholds, where 0.5 is random chance and 1.0 is perfect classification.

### Q12: Why didn't you use Deep Learning (Neural Networks)?
> **Answer:** Deep learning typically requires tens of thousands of samples to avoid catastrophic overfitting. With sample sizes in the hundreds ($N=240$) and thousands of features ($P=1500$), regularized linear models and Random Forests generalize better, train in seconds, and provide transparent feature importances that biologists can interpret.

### Q13: What does the confusion matrix tell us?
> **Answer:** The confusion matrix displays True Positives, True Negatives, False Positives, and False Negatives. In our test set, our model achieved 0 False Negatives (100% recall), confirming that every tumour sample was flagged.

### Q14: How is this code packaged for reproducibility?
> **Answer:** We provide a single bash script (`./run_all.sh`) that reproduces all preprocessing, cross-validation, test set evaluation, plots, and differential expression in seconds. All random seeds are pinned, dependencies are pinned in `requirements.txt`, and automated tests run via `pytest` and GitHub Actions CI.

### Q15: How would you deploy this in a hospital or lab?
> **Answer:** This pipeline is an investigative research tool, not an approved diagnostic medical device. To move towards clinical deployment, it would require CLIA/CAP laboratory validation, external blinded cohort trials, regulatory approval (e.g., FDA 510(k)), and integration into laboratory information systems (LIMS) with strict patient data privacy compliance (HIPAA / GDPR).
