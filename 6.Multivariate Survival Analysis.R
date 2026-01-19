library(dplyr)
library(glmnet)
library(survival)
library(survminer)

set.seed(200)

calc.data <- read.csv("gc_project/cox.data.csv")

drop_cols <- c("roi_id", "time", "status")
feature_cols <- setdiff(colnames(calc.data), drop_cols)

zero_rate <- sapply(calc.data[, feature_cols, drop = FALSE], function(x) mean(x == 0, na.rm = TRUE))
covariates <- names(zero_rate)[zero_rate < 0.50]

x <- as.matrix(calc.data[, covariates, drop = FALSE])
y <- Surv(as.numeric(calc.data$time), as.numeric(calc.data$status))

lasso_fit <- cv.glmnet(x, y, family = "cox", type.measure = "deviance", nfolds = 5)

coef_mat <- coef(lasso_fit, s = lasso_fit$lambda.min)
active_index <- which(as.numeric(coef_mat) != 0)
sig_vars <- rownames(coef_mat)[active_index]

form1 <- as.formula(paste("Surv(time, status) ~", paste(sig_vars, collapse = "+")))
fit1 <- coxph(form1, data = calc.data)

ph <- cox.zph(fit1)
ph_tbl <- ph$table[-nrow(ph$table), , drop = FALSE]

keep_vars <- rownames(ph_tbl)[ph_tbl[, "p"] > 0.05]

form2 <- as.formula(paste("Surv(time, status) ~", paste(keep_vars, collapse = "+")))
fit2 <- coxph(form2, data = calc.data)

summary(fit2)
ggforest(fit2)
