
# install.packages(c("forecast", "tseries", "rugarch", "randomForest", "caret", "zoo", "lubridate"))

library(forecast)
library(tseries)
library(rugarch)
library(randomForest)
library(caret)
library(zoo)
library(lubridate)

if(!require(tseries)){
  install.packages("tseries")
  library(tseries)
}

apple_stock <- read.csv('/Users/s/Downloads/AAPL.csv', header = TRUE, stringsAsFactors = FALSE)

returns <- ts(apple_stock$Adj.Close)

diff_returns <- diff(returns)

plot(diff_returns, main = "First Differenced Returns")
adf.test(diff_returns, alternative = "stationary")

s <- 4

seasonal_diff_returns <- diff(diff_returns, lag = s)

plot(seasonal_diff_returns, main = "Seasonally Differenced Returns")
adf.test(seasonal_diff_returns, alternative = "stationary")

diff_returns <- na.omit(diff_returns)
seasonal_diff_returns <- na.omit(seasonal_diff_returns)


apple_stock <- read.csv('/Users/s/Downloads/AAPL.csv', header = TRUE, stringsAsFactors = FALSE)
apple_stock$Date <- ymd(apple_stock$Date)
apple_stock$Adj.Close <- as.numeric(apple_stock$Adj.Close)



returns <- na.omit(diff(log(apple_stock$Adj.Close))) * 100

returns <- ts(returns)

grid_arima <- expand.grid(p = 0:5, d = 0:2, q = 0:5)

best_arima <- list(aic=Inf, model=NULL)
for(i in 1:nrow(grid_arima)) {
  model <- tryCatch(
    arima(returns, order = c(grid_arima$p[i], grid_arima$d[i], grid_arima$q[i])),
    error=function(e) NULL
  )
  if (!is.null(model) && AIC(model) < best_arima$aic) {
    best_arima$aic <- AIC(model)
    best_arima$model <- model
  }
}
if(!is.null(best_arima$model)) {
  print(summary(best_arima$model)) 
} else {
  cat("No suitable ARIMA model was found.\n")
}

forecast_arima <- forecast(best_arima$model, h = 12)
png("/Users/s/Downloads/forecast_arima.png")
plot(forecast_arima)
dev.off()

grid_garch <- expand.grid(omega = c(1e-6, 1e-5, 1e-4), alpha = c(0.01, 0.05, 0.1), beta = c(0.85, 0.9, 0.95))

best_garch <- list(logLik=-Inf, model=NULL)
for(i in 1:nrow(grid_garch)) {
  spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), 
                     mean.model = list(armaOrder = c(2, 1), include.mean = TRUE), 
                     distribution.model = "sstd")
  garch_fit <- tryCatch(
    ugarchfit(spec = spec, data = returns, solver.control = list(trace=0)),
    error=function(e) NULL
  )
  if (!is.null(garch_fit)) {
    criteria <- infocriteria(garch_fit)
    if (!is.null(criteria) && !is.na(criteria["logLik"]) && criteria["logLik"] > best_garch$logLik) {
      best_garch$logLik <- criteria["logLik"]
      best_garch$model <- garch_fit
    }
  }
}
if(!is.null(best_garch$model)) {
  print(summary(best_garch$model))
} else {
  cat("No suitable GARCH model was found.\n")
}


library(forecast)

seasonal_period <- 12  

sarima_model <- auto.arima(returns, seasonal = TRUE, D = 1, max.P = 2, max.Q = 2, max.D = 1, period = seasonal_period)


summary(sarima_model)


n_forecast <- 12
forecast_sarima <- forecast(sarima_model, h = n_forecast)

plot(forecast_sarima)




set.seed(123)

grid_rf <- expand.grid(mtry = 1, splitrule = c("variance"), min.node.size = c(5, 10))

lag_returns <- lag(returns, -1) 
data_rf <- data.frame(Returns = returns[-length(returns)], LagReturns = lag_returns[-1])
data_rf <- na.omit(data_rf)  

train_indices <- sample(seq_len(nrow(data_rf)), size = floor(0.8 * nrow(data_rf)))
train_data <- data_rf[train_indices, ]
test_data <- data_rf[-train_indices, ]

train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

tryCatch({
  rf_model <- train(Returns ~ LagReturns, data = train_data, method = "ranger",
                    tuneGrid = grid_rf, trControl = train_control)
  # 模型最优参数
  cat("Best Random Forest Model:\n")
  print(rf_model$bestTune)  # 打印最佳参数

  pred_rf <- predict(rf_model, test_data)
  cat("Random Forest Test Set RMSE:", RMSE(pred_rf, test_data$Returns))

  png("/Users/s/Downloads/forecast_rf.png")
  plot(test_data$Returns, type = 'l', col = 'blue', main = "Random Forest Predictions vs Actual")
  lines(pred_rf, type = 'l', col = 'red')
  dev.off()
}, error=function(e){
  cat("Error in Random Forest training: ", e$message, "\n")
})

checkresiduals(best_arima$model)
