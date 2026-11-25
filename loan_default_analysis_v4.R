# Bayesian Loan Default Analysis
# The project studies the association between borrower and loan characteristics
# and loan default using frequentist and Bayesian logistic regression.
# *** Dataset: https://www.kaggle.com/datasets/yasserh/loan-default-dataset

# Part 0: setup

# Install missing packages once, in a fresh R session:
# install.packages(c("MCMCpack", "HDInterval"))
# install.packages(c("ggplot2", "corrplot", "coda", "reshape2"))

library(MCMCpack)
library(HDInterval)
library(ggplot2)
library(corrplot)
library(coda)
library(reshape2)

set.seed(123)
stopifnot(file.exists("Loan_Default.csv"))
loan.default = read.csv("Loan_Default.csv")
predictor.names = c("Credit_Score", "income", "LTV", "dtir1", "loan_amount")
selected.columns = c(predictor.names, "Status")
stopifnot(all(selected.columns %in% names(loan.default)), nrow(loan.default) > 0)

# Part 1: data cleaning

# Always restart from the original dataset.
loan.data = loan.default[, selected.columns]
stopifnot(all(sapply(loan.data, is.numeric)),
          all(na.omit(loan.data$Status) %in% c(0, 1)))

# Missing and non-finite values
for (column in predictor.names) {
  loan.data[[column]][!is.finite(loan.data[[column]])] = NA
}
missing.summary = data.frame(Variable = selected.columns,
                             Missing = colSums(is.na(loan.data)),
                             Percentage = 100 * colMeans(is.na(loan.data)),
                             row.names = NULL)
print(missing.summary)

# Remove missing responses and impute the same predictors as in V3.
loan.data = loan.data[!is.na(loan.data$Status), ]
impute.columns = c("dtir1", "LTV", "loan_amount")
imputation.summary = data.frame(Variable = impute.columns,
                                Imputed = colSums(is.na(loan.data[, impute.columns])),
                                Median = sapply(loan.data[, impute.columns], median, na.rm = TRUE),
                                row.names = NULL)
stopifnot(all(is.finite(imputation.summary$Median)))
for (column in impute.columns) {
  loan.data[[column]] = replace(loan.data[[column]], is.na(loan.data[[column]]),
                                median(loan.data[[column]], na.rm = TRUE))
}
stopifnot(!anyNA(loan.data[, impute.columns]))
print(imputation.summary)

# Remove observations with remaining missing values (mainly income).
loan.data = na.omit(loan.data)
stopifnot(nrow(loan.data) > 1, all(is.finite(as.matrix(loan.data))),
          length(unique(loan.data$Status)) == 2)
cleaning.summary = data.frame(Original = nrow(loan.default),
                              Removed = nrow(loan.default) - nrow(loan.data),
                              Retained = nrow(loan.data))
print(cleaning.summary)
loan.before.capping = loan.data

# IQR capping: keep observations, limit extreme predictor values.
for (column in predictor.names) {
  quartiles = quantile(loan.data[[column]], c(0.25, 0.75))
  interquartile.range = IQR(loan.data[[column]])
  lower.bound = quartiles[1] - 1.5 * interquartile.range
  upper.bound = quartiles[2] + 1.5 * interquartile.range
  if (interquartile.range > 0) {
    loan.data[[column]] = pmax(pmin(loan.data[[column]], upper.bound), lower.bound)
  }
}
outlier.summary = data.frame(Variable = predictor.names,
                             Capped = colSums(loan.before.capping[, predictor.names] !=
                                              loan.data[, predictor.names]), row.names = NULL)
print(outlier.summary)
stopifnot(nrow(loan.data) == nrow(loan.before.capping),
          all(loan.data$Status == loan.before.capping$Status))
loan.before.scaling = loan.data

# Standardize predictors using the built-in R function.
stopifnot(all(sapply(loan.data[, predictor.names], sd) > 0))
loan.data[, predictor.names] = scale(loan.data[, predictor.names])
standardization.summary = data.frame(Variable = predictor.names,
                                     Mean = colMeans(loan.data[, predictor.names]),
                                     SD = sapply(loan.data[, predictor.names], sd), row.names = NULL)
stopifnot(all(abs(standardization.summary$Mean) < 1e-8),
          all(abs(standardization.summary$SD - 1) < 1e-8))
print(standardization.summary)
cat("Sanity check: data cleaning and standardization OK.\n")

# Part 2: exploratory data analysis

class.summary = data.frame(Status = c("Non-default", "Default"),
                           Count = as.vector(table(factor(loan.data$Status, levels = c(0, 1)))))
class.summary$Percentage = 100 * class.summary$Count / sum(class.summary$Count)
raw.summary = summary(loan.before.capping)
cleaned.summary = summary(loan.before.scaling)
correlation.matrix = cor(loan.before.capping[, predictor.names])
print(class.summary)
print(raw.summary)
print(round(correlation.matrix, 3))
p.class = ggplot(class.summary, aes(Status, Count, fill = Status)) +
  geom_col(show.legend = FALSE) + theme_minimal() + labs(title = "Loan status distribution")
raw.long = melt(loan.before.capping[, predictor.names])
clean.long = melt(loan.before.scaling[, predictor.names])
scaled.long = melt(loan.data[, predictor.names])
p.before = ggplot(raw.long, aes(x = variable, y = value)) + geom_boxplot() +
  facet_wrap(~ variable, scales = "free", ncol = 2) + theme_minimal() +
  labs(title = "Before outlier capping", x = NULL, y = NULL)
p.after = ggplot(clean.long, aes(x = variable, y = value)) + geom_boxplot() +
  facet_wrap(~ variable, scales = "free", ncol = 2) + theme_minimal() +
  labs(title = "After outlier capping", x = NULL, y = NULL)
p.standardized = ggplot(scaled.long, aes(value)) + geom_histogram(bins = 30) +
  facet_wrap(~ variable, scales = "free_y", ncol = 2) + theme_minimal() +
  labs(title = "Standardized predictors", x = "Z-score", y = "Count")

# Part 3: frequentist logistic regression

model.formula = Status ~ Credit_Score + income + LTV + dtir1 + loan_amount
logit.model = glm(model.formula, data = loan.data, family = binomial)
stopifnot(logit.model$converged, all(is.finite(coef(logit.model))))
frequentist.results = summary(logit.model)$coefficients
print(frequentist.results)

# Part 4: Bayesian logistic regression

# Independent N(0, 1) priors for all coefficients, including the intercept.
# This replaces the hierarchical Gamma precision prior in V3.
# MCMCpack uses a covariance-aware random walk Metropolis proposal.
# mcmc counts retained iterations after burn-in when thin = 1.

burn.in.period = 2000
n.iter = 10000
proposal.scale = 1.1
starting.values = coef(logit.model)
starting.sd = sqrt(diag(vcov(logit.model)))

chain.one = MCMClogit(model.formula, data = loan.data, b0 = 0, B0 = 1,
                     burnin = burn.in.period, mcmc = n.iter, thin = 1,
                     tune = proposal.scale, seed = 123, verbose = 1000,
                     beta.start = as.numeric(starting.values - 3 * starting.sd))
chain.two = MCMClogit(model.formula, data = loan.data, b0 = 0, B0 = 1,
                     burnin = burn.in.period, mcmc = n.iter, thin = 1,
                     tune = proposal.scale, seed = 456, verbose = 1000,
                     beta.start = as.numeric(starting.values + 3 * starting.sd))

samples.mcmc = mcmc.list(chain.one, chain.two)
posterior.samples = rbind(as.matrix(chain.one), as.matrix(chain.two))
stopifnot(nrow(posterior.samples) == 2 * n.iter,
          all(is.finite(posterior.samples)), all(apply(posterior.samples, 2, sd) > 0),
          identical(colnames(posterior.samples), names(coef(logit.model))))
cat("Sanity check: posterior samples OK.\n")

# Part 5: posterior analysis

posterior.means = colMeans(posterior.samples)
credible.intervals = t(apply(posterior.samples, 2, quantile, c(0.025, 0.975)))
hdi.bounds = t(hdi(posterior.samples, credMass = 0.95))
posterior.results = data.frame(Parameter = colnames(posterior.samples),
                                Mean = posterior.means,
                                SD = apply(posterior.samples, 2, sd),
                                Credible_Lower = credible.intervals[, 1],
                                Credible_Upper = credible.intervals[, 2],
                                HDI_Lower = hdi.bounds[, 1], HDI_Upper = hdi.bounds[, 2],
                                row.names = NULL)
stopifnot(all(hdi.bounds[, 1] <= hdi.bounds[, 2]))
print(posterior.results)
model.comparison = data.frame(Parameter = names(coef(logit.model)),
                               Frequentist = as.numeric(coef(logit.model)),
                               Bayesian = as.numeric(posterior.means))
print(model.comparison)

# Odds ratios are for a one-SD increase, holding other predictors fixed.
odds.results = data.frame(Parameter = posterior.results$Parameter,
                          Median_OR = apply(exp(posterior.samples), 2, median),
                          Lower = exp(credible.intervals[, 1]),
                          Upper = exp(credible.intervals[, 2]), row.names = NULL)
odds.results = odds.results[odds.results$Parameter != "(Intercept)", ]
posterior.correlation = cor(posterior.samples)
samples.long = melt(as.data.frame(posterior.samples),
                    measure.vars = colnames(posterior.samples),
                    variable.name = "Parameter", value.name = "Value")
p.posterior = ggplot(samples.long, aes(Value)) +
  geom_density(fill = "skyblue", alpha = 0.5) +
  geom_vline(data = posterior.results, aes(xintercept = HDI_Lower),
             linetype = "dashed", color = "red") +
  geom_vline(data = posterior.results, aes(xintercept = HDI_Upper),
             linetype = "dashed", color = "red") +
  facet_wrap(~ Parameter, scales = "free", ncol = 2) + theme_minimal() +
  labs(title = "Posterior distributions with 95% HDI", x = "Coefficient", y = "Density")
p.comparison = ggplot(melt(model.comparison, id.vars = "Parameter"),
                      aes(Parameter, value, color = variable)) +
  geom_point(position = position_dodge(width = 0.4), size = 2) + coord_flip() +
  theme_minimal() + labs(title = "Frequentist and Bayesian estimates", y = "Coefficient", color = "Model")
p.odds = ggplot(odds.results, aes(Parameter, Median_OR)) +
  geom_pointrange(aes(ymin = Lower, ymax = Upper)) + geom_hline(yintercept = 1, linetype = "dashed") +
  coord_flip() + theme_minimal() + labs(title = "Posterior odds ratios (95% credible intervals)", y = "Odds ratio")

# Part 6: MCMC diagnostics

# Compare two chains; inspect each chain separately for Geweke and ACF.
effective.sample.size = effectiveSize(samples.mcmc)
geweke.one = geweke.diag(chain.one)$z
geweke.two = geweke.diag(chain.two)$z
rhat = gelman.diag(samples.mcmc, autoburnin = FALSE, multivariate = FALSE)$psrf[, 1]
diagnostic.summary = data.frame(Parameter = names(effective.sample.size),
                                 ESS = as.numeric(effective.sample.size),
                                 Geweke_1 = as.numeric(geweke.one),
                                 Geweke_2 = as.numeric(geweke.two),
                                 Rhat = as.numeric(rhat))
print(diagnostic.summary)
if (any(!is.finite(as.matrix(diagnostic.summary[, -1]))) ||
    any(diagnostic.summary$ESS < 400) || any(diagnostic.summary$Rhat > 1.01)) {
  warning("Review ESS, Rhat and trace plots before interpreting results.")
}
if (any(abs(c(geweke.one, geweke.two)) > 1.96, na.rm = TRUE)) {
  warning("Geweke flags possible non-stationarity; inspect both chains.")
}
# Acceptance rates are printed by MCMClogit in the console.
# Diagnostic thresholds are guidelines, not proof of convergence.
mcmc.settings = data.frame(Chains = 2, Burn_in_per_chain = burn.in.period,
                           Retained_per_chain = n.iter, Thinning = 1,
                           Tune = proposal.scale, Prior_SD = 1)

# Part 7: results and reproducibility

# This is an inferential academic exercise, not a validated predictive model.
# Associations are conditional on the other predictors and the chosen cleaning.
# Median imputation and IQR capping may influence the estimates.
# Inspect diagnostics before using posterior summaries.
print(mcmc.settings)
print(sessionInfo())

# Part 8: presentation material

# Tables: run any line below to recall the corresponding result.
print(cleaning.summary)
print(missing.summary)
print(imputation.summary)
print(class.summary)
print(outlier.summary)
print(raw.summary)
print(cleaned.summary)
print(standardization.summary)
print(frequentist.results)
print(mcmc.settings)
print(posterior.results)
print(model.comparison)
print(odds.results)
print(diagnostic.summary)

# Graphs: run individual lines to display or export plots in RStudio.
print(p.class)
print(p.before)
print(p.after)
print(p.standardized)
corrplot(correlation.matrix, method = "color", type = "upper", addCoef.col = "black", tl.col = "black")
corrplot(posterior.correlation, method = "color", type = "upper", addCoef.col = "black", tl.col = "black")
print(p.posterior)
print(p.comparison)
print(p.odds)
traceplot(samples.mcmc)
acfplot(chain.one)
acfplot(chain.two)
