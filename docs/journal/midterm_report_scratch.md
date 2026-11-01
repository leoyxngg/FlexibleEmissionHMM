### Evaluation on S&P 500

### Feature Engineering

We computed these features given the log return. These factors were showed to be useful for regime detection (cite something).

#### PCA

This is a wildly used technique to reduce the dimention and the noise in the data (cite somethign). We know there are noises in the financial market. The reduction in the dimention would help our model to not fit the noise to improve the model training proces.

### Experiment

In the actual finance world, people tend to care about the sharpe ratio and max drawdown which are the metrics of a strategy. We want to present our experiments with the simple signal driven HMM strategy and compare with the normal Gaussian HMM models.

In this paper, they mentioned student-t with PCA dimention reduction have a better persistent HMM. They ran the experiment on s&p 500. Their experiment setting is for every prediction t+1 it fits a new HMM and PCA based on the lag t-10 years features, and made decision based on the 12-day rolling median to decide to all-in or hold cash. However, refitting HMM every t are computationally heavy and the HMM distribution would not change to much by replacing the data with t-10years with t+1 in the next iteration.

In our experiement, the time period was between 2004 and 2025 which aross multiple economic crises. We do a expanding window to predict the t+1 log return given the log return starts at 2004 Jan 1st. The HMM and PCA are refilled every 63 trading days and run 3 more EM steps when it's not refitting day and expand the training window instead of sliding the window. We compare our results with buy and hold strategy and MA50_200 which all in sp500 when the SMA50 is above the SMA 200 otherwise hold cash. The results show that student-T distribution and GMM distribution with 3 components on max drawdown and sharpe ratio perform better than the baseline buy and hold and ma50_200 strategies. Base on the heatmap, we can see that the next day return conditional on the model bull or bear does what we wants.

#### Case study the distrubtion before and after the financial crisis

We can see that the distribution dramatically drift after the financial crisis, we can justify a HMM reacts to the fast pace regime speed base. We can see that student-T reacts that the bear market is done and show both the persistent and accuracy to detch the market after the crisis which boosted its sharpe ratio because 2009 was a uphill market. We also compare with other model and we can see the difference between each underlying distribution. A key finding is that the bull mark and bear market should have two distribution and one should look more normal where other is skewed and has lower variance.

### GMM with different number of components

Base on our another experiment that runs GMM with more number of components, we can see that the performance difference are quite large. We can furthere investigate this base on the held-out-test (cite paper). The idea is to compare the test log score for models and see which model the data flavour. (Math formula). We see that GMM with 3 components have the best score. We also furthere inverstigate the skewness and how each components contribute to the model. From this interval plot, one or two components of GMM 4 or 5 fit into the long tail with relatively larege variance. It overfits the data and we can also justify by the convergence rate. We give each model 100 iterations bugdet and the mean is 50 iterations to converge. GMM 4 and 5 have the lower convergence rate.
