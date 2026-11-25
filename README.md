# Bayesian Loan Default Analysis

## Overview

This academic project investigates the association between borrower and loan characteristics and loan default using frequentist and Bayesian logistic regression in R. The aim is to compare coefficient estimates, quantify posterior uncertainty and assess the quality of MCMC sampling.

## Dataset

Source: [Loan Default Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset).

The original dataset contains 148,670 observations. After preprocessing, 139,520 observations are retained, including 35,400 defaults (25.37%). The binary response variable is `Status` (0 = non-default, 1 = default).

The five predictors are credit score, income, loan-to-value ratio (LTV), debt-to-income ratio (DTIR1) and loan amount.

## Methodology

- Data cleaning: median imputation for missing LTV and DTIR1 values, removal of remaining incomplete observations, IQR-based outlier capping and predictor standardization.
- Exploratory data analysis: descriptive statistics, class distribution, boxplots, histograms and correlation analysis.
- Frequentist logistic regression using `glm()` as a benchmark.
- Bayesian logistic regression using `MCMCpack::MCMClogit()` with independent standard normal priors on all coefficients, including the intercept.
- Posterior analysis using posterior means, standard deviations, 95% credible intervals, highest density intervals (HDIs) and odds ratios.
- MCMC diagnostics using trace plots, autocorrelation, effective sample size, Geweke diagnostics and the Gelman–Rubin diagnostic.

## MCMC configuration

Two Random Walk Metropolis chains are run with different starting values and seeds (123 and 456). Each chain uses 2,000 burn-in iterations and 10,000 retained iterations, without thinning. The tuning parameter is 1.1.

## Results

| Parameter | Frequentist estimate | Posterior mean | 95% credible interval |
|---|---:|---:|---|
| Intercept | -1.1214 | -1.1214 | [-1.1336, -1.1088] |
| Credit Score | 0.0099 | 0.0102 | [-0.0020, 0.0225] |
| Income | -0.3031 | -0.3032 | [-0.3208, -0.2858] |
| LTV | 0.2340 | 0.2340 | [0.2217, 0.2465] |
| DTIR1 | 0.0998 | 0.1000 | [0.0872, 0.1135] |
| Loan Amount | 0.0531 | 0.0533 | [0.0370, 0.0697] |

The Bayesian estimates are close to the frequentist estimates. Income is negatively associated with default, while LTV, DTIR1 and loan amount are positively associated with default, holding the other predictors fixed. The credible interval for credit score includes zero.

The acceptance rates are 22.84% and 22.73%. Effective sample sizes range from approximately 972 to 1,061, and all Gelman–Rubin point estimates are below 1.01. One of the twelve Geweke Z-scores falls outside the reference interval of ±1.96 (credit score, first chain: 2.168). Overall, the diagnostics are encouraging, although they do not establish convergence conclusively.

## Reproducing the analysis

1. Download and extract `Loan_Default.csv` from the Kaggle dataset linked above.
2. Place the CSV in the same directory as `loan_default_analysis_v4.R`.
3. Open R or RStudio, set the working directory to that folder and install any missing packages:

```r
install.packages(c("MCMCpack", "HDInterval", "ggplot2", "corrplot", "coda", "reshape2"))
```

4. Run the script:

```r
source("loan_default_analysis_v4.R")
```

The script includes sanity checks, diagnostic summaries and a final section for displaying tables and plots.

Tested environment: R 4.3.0 on Windows, MCMCpack 1.7-0, HDInterval 0.2.4, ggplot2 3.5.1, corrplot 0.95, coda 0.19-4.1 and reshape2 1.4.4. Package versions compatible with your R installation may be required.

## Limitations

This is an academic inference exercise, not a validated predictive credit risk model. The results describe conditional associations, not causal effects. Median imputation, removal of incomplete observations and outlier capping may affect the estimates. No out-of-sample predictive evaluation is performed.
