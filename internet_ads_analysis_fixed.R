# ============================================================
# DATA ANALYSIS: INTERNET ADVERTISEMENTS DATASET
# UCI Machine Learning Repository
# ============================================================

# ============================================================
# 0. INSTALL AND LOAD LIBRARIES
# ============================================================
# install.packages(c("tidyverse", "caret", "randomForest", "ROCR",
#                   "corrplot", "ggplot2", "pROC", "e1071", "dplyr"))

library(tidyverse)
library(caret)
library(randomForest)
library(ROCR)
library(ggplot2)
library(pROC)
library(dplyr)

# ============================================================
# 1. READ AND OVERVIEW DATA
# ============================================================

# NOTE: Missing values in this CSV are encoded as "   ?" (3 spaces + ?).
# na.strings must include this exact string. "NA" and "" are also listed
# defensively. header = TRUE because add.csv has a header row.
# IMPORTANT: This file has ~903 rows with missing height/width/aratio (~27.5%),
# NOT the ~28 rows (~0.85%) stated in the report. The report's Table 2 is wrong.
df_raw <- read.csv("add.csv", header = TRUE,
                   na.strings = c("?", " ?", "  ?", "   ?", "NA", ""),
                   stringsAsFactors = FALSE)

# add.csv has a row-index column as the first column — drop it.
df_raw <- df_raw[, -1]

# Assign column names: height, width, aratio, feat_1..feat_N, class
n_cols <- ncol(df_raw)
colnames(df_raw) <- c("height", "width", "aratio",
                      paste0("feat_", 1:(n_cols - 4)),
                      "class")

df_raw$height <- as.numeric(df_raw$height)
df_raw$width  <- as.numeric(df_raw$width)
df_raw$aratio <- as.numeric(df_raw$aratio)

# Convert class label to factor
df_raw$class <- factor(trimws(df_raw$class),
                       levels = c("nonad.", "ad."),
                       labels = c("nonad", "ad"))

# FIX 1: Corrected comment — actual CSV has 0 rows with missing class, not 1.
# The filter is kept as a safe guard but will not drop any rows.
df_raw <- df_raw[!is.na(df_raw$class), ]

cat("=== DATA OVERVIEW ===\n")
cat("Number of rows:", nrow(df_raw), "\n")
cat("Number of columns:", ncol(df_raw), "\n")
cat("\nClass label distribution:\n")
print(table(df_raw$class))
cat("\nPercentage (%):\n")
print(round(prop.table(table(df_raw$class)) * 100, 2))

cat("\nSummary of 3 continuous variables (before imputation):\n")
print(summary(df_raw[, c("height", "width", "aratio")]))

cat("\nMissing value count (NA) by continuous variable:\n")
cat("height:", sum(is.na(df_raw$height)), "\n")
cat("width: ", sum(is.na(df_raw$width)),  "\n")
cat("aratio:", sum(is.na(df_raw$aratio)), "\n")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

cat("\n=== PREPROCESSING ===\n")

df <- df_raw  # work on a copy

# Impute missing values in continuous vars with median (robust to outliers)
df$height[is.na(df$height)] <- median(df$height, na.rm = TRUE)
df$width[is.na(df$width)]   <- median(df$width,  na.rm = TRUE)
df$aratio[is.na(df$aratio)] <- median(df$aratio, na.rm = TRUE)

# Convert all binary feature variables to numeric and impute NAs with 0
binary_cols <- grep("^feat_", colnames(df), value = TRUE)
df[binary_cols] <- lapply(df[binary_cols], function(x) {
  x <- as.numeric(x)
  x[is.na(x)] <- 0
  return(x)
})

cat("After preprocessing - Total remaining NAs:", sum(is.na(df)), "\n")

# Single unified train/test split used for BOTH models.
# Using one split ensures both models are evaluated on identical test data,
# making the comparison in the results table valid.
set.seed(42)
trainIndex <- createDataPartition(df$class, p = 0.7, list = FALSE)
train_df <- df[trainIndex, ]
test_df  <- df[-trainIndex, ]

cat("Training set:", nrow(train_df), "samples\n")
cat("Test set:    ", nrow(test_df),  "samples\n")

# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

cat("\n=== DESCRIPTIVE STATISTICS ===\n")

desc_stats <- df %>%
  group_by(class) %>%
  summarise(
    n           = n(),
    height_mean = round(mean(height), 2),
    height_sd   = round(sd(height),   2),
    width_mean  = round(mean(width),  2),
    width_sd    = round(sd(width),    2),
    aratio_mean = round(mean(aratio), 2),
    aratio_sd   = round(sd(aratio),   2),
    .groups = "drop"
  )
print(desc_stats)

# Plot 1: Class label distribution
ggplot(df, aes(x = class, fill = class)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Figure 1: Class Distribution (ad vs nonad)",
       x = "Class Label", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("plot1_class_distribution.png", width = 6, height = 4)

# Plot 2: Boxplot of height by class
# coord_cartesian caps y-axis at a readable range.
# Without this, extreme outliers stretch the axis to ~3000px.
ggplot(df, aes(x = class, y = height, fill = class)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_cartesian(ylim = c(0, 600)) +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Figure 2: Distribution of Height by Class (y-axis capped at 600px)",
       x = "Class Label", y = "Height (pixels)") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("plot2_height_boxplot.png", width = 6, height = 4)

# Plot 3: Scatter plot of width vs height
ggplot(df, aes(x = width, y = height, color = class)) +
  geom_point(alpha = 0.4, size = 1.5) +
  scale_color_manual(values = c("steelblue", "tomato")) +
  labs(title = "Figure 3: Width vs Height by Class",
       x = "Width (pixels)", y = "Height (pixels)", color = "Class") +
  theme_minimal() +
  coord_cartesian(xlim = c(0, 650), ylim = c(0, 500))
ggsave("plot3_width_height_scatter.png", width = 7, height = 5)

# Plot 4: Histogram of aspect ratio by class
ggplot(df, aes(x = aratio, fill = class)) +
  geom_histogram(bins = 40, alpha = 0.7, position = "identity") +
  coord_cartesian(xlim = c(0, 25)) +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Figure 4: Distribution of Aspect Ratio by Class",
       x = "Aspect Ratio (width/height)", y = "Frequency", fill = "Class") +
  theme_minimal()
ggsave("plot4_aratio_histogram.png", width = 7, height = 4)

# ============================================================
# 4. HYPOTHESIS TESTING (INFERENTIAL STATISTICS - PART 1)
# ============================================================

cat("\n=== HYPOTHESIS TESTING ===\n")
cat("NOTE: Actual results differ from report Table 5 because the report\n")
cat("      used approximate values from a different data version (28 missing\n")
cat("      rows vs actual 903). Actual results from this CSV are shown below.\n\n")

# Test 1: Welch t-test for height
cat("\n--- Test 1: t-test for HEIGHT ---\n")
cat("H0: Mean height of ad = nonad\n")
cat("H1: Mean height of ad != nonad\n")
t_height <- t.test(height ~ class, data = df)
print(t_height)
cat("Conclusion: p-value =", round(t_height$p.value, 6),
    ifelse(t_height$p.value < 0.05,
           "=> Reject H0. Statistically significant difference.\n",
           "=> Fail to reject H0.\n"))

# Test 2: Welch t-test for width
cat("\n--- Test 2: t-test for WIDTH ---\n")
cat("H0: Mean width of ad = nonad\n")
cat("H1: Mean width of ad != nonad\n")
t_width <- t.test(width ~ class, data = df)
print(t_width)
cat("Conclusion: p-value =", round(t_width$p.value, 6),
    ifelse(t_width$p.value < 0.05,
           "=> Reject H0. Statistically significant difference.\n",
           "=> Fail to reject H0.\n"))

# Test 3: Welch t-test for aratio
cat("\n--- Test 3: t-test for ARATIO ---\n")
cat("H0: Mean aratio of ad = nonad\n")
cat("H1: Mean aratio of ad != nonad\n")
t_aratio <- t.test(aratio ~ class, data = df)
print(t_aratio)
cat("Conclusion: p-value =", round(t_aratio$p.value, 6),
    ifelse(t_aratio$p.value < 0.05,
           "=> Reject H0. Statistically significant difference.\n",
           "=> Fail to reject H0.\n"))

# Plot 5: Mean comparison of 3 continuous variables
# facet_wrap gives each variable its own y-axis so aratio is not dwarfed by width.
means_df <- df %>%
  group_by(class) %>%
  summarise(height = mean(height), width = mean(width), aratio = mean(aratio),
            .groups = "drop") %>%
  pivot_longer(-class, names_to = "variable", values_to = "mean_value")

ggplot(means_df, aes(x = class, y = mean_value, fill = class)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ variable, scales = "free_y") +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(title = "Figure 5: Mean Values of Continuous Variables by Class",
       x = "Class", y = "Mean Value", fill = "Class") +
  theme_minimal()
ggsave("plot5_means_comparison.png", width = 9, height = 4)

# ============================================================
# 5. LOGISTIC REGRESSION (INFERENTIAL STATISTICS - PART 2)
# ============================================================

cat("\n=== LOGISTIC REGRESSION ===\n")
cat("Objective: Predict probability an image is an advertisement using 3 continuous vars\n\n")

logit_model <- glm(class ~ height + width + aratio,
                   data   = train_df,
                   family = binomial(link = "logit"))

cat("\nLogistic Regression model results:\n")
print(summary(logit_model))

cat("\nOdds Ratios:\n")
print(round(exp(coef(logit_model)), 4))

cat("\n95% Confidence Intervals for Odds Ratios:\n")
print(round(exp(confint(logit_model)), 4))

# Predictions on test set
logit_pred_prob  <- predict(logit_model, newdata = test_df, type = "response")
logit_pred_class <- factor(ifelse(logit_pred_prob > 0.5, "ad", "nonad"),
                           levels = c("nonad", "ad"))

cat("\nConfusion Matrix - Logistic Regression:\n")
cm_logit <- confusionMatrix(logit_pred_class, test_df$class, positive = "ad")
print(cm_logit)

roc_logit <- roc(test_df$class, logit_pred_prob, levels = c("nonad", "ad"))
cat("\nAUC - Logistic Regression:", round(auc(roc_logit), 4), "\n")

# Plot 6: ROC curve - Logistic Regression
png("plot6_roc_logistic.png", width = 600, height = 500)
plot(roc_logit, col = "steelblue", lwd = 2,
     main = paste("Figure 6: ROC Curve - Logistic Regression\nAUC =",
                  round(auc(roc_logit), 4)))
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# ============================================================
# 6. RANDOM FOREST (INFERENTIAL STATISTICS - PART 3)
# ============================================================

cat("\n=== RANDOM FOREST ===\n")
cat("Objective: Classify ad/nonad using the full feature set\n")

# FIX 2: Use ALL available feature columns (feat_1 through feat_1555) instead
# of arbitrarily truncating to feat_1:50. The original code silently discarded
# ~1,500 binary features, significantly reducing model performance.
# The seed is set once here for RF reproducibility; the earlier set.seed(42)
# at the split is separate and not reset here.
set.seed(42)
all_feats       <- c("height", "width", "aratio",
                     grep("^feat_", colnames(train_df), value = TRUE))
available_feats <- intersect(all_feats, colnames(train_df))

cat("Total features used in Random Forest:", length(available_feats), "\n")

rf_train <- train_df[, c(available_feats, "class")]
rf_test  <- test_df[,  c(available_feats, "class")]

rf_model <- randomForest(
  class ~ .,
  data       = rf_train,
  ntree      = 200,
  mtry       = floor(sqrt(length(available_feats))),
  importance = TRUE
)

cat("\nRandom Forest model summary:\n")
print(rf_model)

rf_pred      <- predict(rf_model, newdata = rf_test)
rf_pred_prob <- predict(rf_model, newdata = rf_test, type = "prob")[, "ad"]

cat("\nConfusion Matrix - Random Forest:\n")
cm_rf <- confusionMatrix(rf_pred, rf_test$class, positive = "ad")
print(cm_rf)

roc_rf <- roc(rf_test$class, rf_pred_prob, levels = c("nonad", "ad"))
cat("\nAUC - Random Forest:", round(auc(roc_rf), 4), "\n")

# Plot 7: Feature Importance (top 20 by Mean Decrease in Accuracy)
importance_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  MeanDecAcc = importance(rf_model)[, "MeanDecreaseAccuracy"]
) %>%
  arrange(desc(MeanDecAcc)) %>%
  head(20)

ggplot(importance_df, aes(x = reorder(Feature, MeanDecAcc), y = MeanDecAcc)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Figure 7: Top 20 Feature Importance - Random Forest",
       x = "Feature", y = "Mean Decrease in Accuracy") +
  theme_minimal()
ggsave("plot7_feature_importance.png", width = 7, height = 6)

# Plot 8: ROC curve comparison — both models evaluated on the same test set
png("plot8_roc_comparison.png", width = 700, height = 550)
plot(roc_logit, col = "steelblue", lwd = 2,
     main = "Figure 8: ROC Curve Comparison")
plot(roc_rf, col = "tomato", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Logistic (AUC =", round(auc(roc_logit), 3), ")"),
                  paste("Random Forest (AUC =", round(auc(roc_rf),    3), ")")),
       col = c("steelblue", "tomato"), lwd = 2)
dev.off()

# ============================================================
# 7. RESULTS SUMMARY
# ============================================================

cat("\n=== MODEL RESULTS SUMMARY ===\n")
cat("NOTE: Logistic Regression uses 3 continuous predictors (height, width, aratio).\n")
cat("      Random Forest uses all", length(available_feats), "features.\n")
cat("      Both models are evaluated on the same held-out test set.\n\n")

results <- data.frame(
  Model       = c("Logistic Regression", "Random Forest"),
  Accuracy    = c(round(cm_logit$overall["Accuracy"],    4),
                  round(cm_rf$overall["Accuracy"],       4)),
  Sensitivity = c(round(cm_logit$byClass["Sensitivity"], 4),
                  round(cm_rf$byClass["Sensitivity"],    4)),
  Specificity = c(round(cm_logit$byClass["Specificity"], 4),
                  round(cm_rf$byClass["Specificity"],    4)),
  AUC         = c(round(auc(roc_logit), 4),
                  round(auc(roc_rf),    4))
)
print(results)

cat("\n=== COMPARISON: ACTUAL vs REPORTED ===\n")
cat(sprintf("%-22s | %-8s %-8s %-8s %-8s\n", "Model", "Accuracy", "Sensitiv", "Specific", "AUC"))
cat(sprintf("%-22s | %-8s %-8s %-8s %-8s\n", "Logistic (actual)",
  sprintf("%.1f%%", 100*cm_logit$overall["Accuracy"]),
  sprintf("%.1f%%", 100*cm_logit$byClass["Sensitivity"]),
  sprintf("%.1f%%", 100*cm_logit$byClass["Specificity"]),
  sprintf("%.3f",   auc(roc_logit))))
cat(sprintf("%-22s | %-8s %-8s %-8s %-8s\n", "Logistic (reported)",
  "83%","62%","88%","0.83"))
cat(sprintf("%-22s | %-8s %-8s %-8s %-8s\n", "RandForest (actual)",
  sprintf("%.1f%%", 100*cm_rf$overall["Accuracy"]),
  sprintf("%.1f%%", 100*cm_rf$byClass["Sensitivity"]),
  sprintf("%.1f%%", 100*cm_rf$byClass["Specificity"]),
  sprintf("%.3f",   auc(roc_rf))))
cat(sprintf("%-22s | %-8s %-8s %-8s %-8s\n", "RandForest (reported)",
  "96%","91%","97%","0.98"))

cat("\nData source: https://www.kaggle.com/datasets/uciml/internet-advertisements-data-set\n")
cat("R version:", R.version$version.string, "\n")
cat("Analysis complete!\n")
