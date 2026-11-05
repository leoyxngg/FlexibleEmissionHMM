[mwshumway.github.io/data/market_regime_id.pdf](https://mwshumway.github.io/data/market_regime_id.pdf)

In this paper, they mentioned student-t with PCA dimention reduction have a better persistent HMM. They ran the experiment on s&p 500. Their experiment setting is for every prediction t+1 it fits a new HMM and PCA based on the lag t-10 years features, and made decision based on the 12-day rolling median to decide to all-in or hold cash.

![1792281615175](image/Oct17/1792281615175.png)

![1792281624413](image/Oct17/1792281624413.png)

I tried to replicate the experiment setting, however, the refitting for every new trading day is too computationally expensive. I refit PCA and HMM every 63 trading days. For non refitting days, I ran 3 more EM step for my algorithm. This dramatically reduce the computation. Also, instead of the trailing 10 year fitting window, I only append new day but not drop the old data, so it that last few fit, it inlcudes more than 20 years of data. In this experiment, I was focusing on two regimes model.

![1792282957618](image/Oct17/1792282957618.png)

![1792282990969](image/Oct17/1792282990969.png)

![1792283075738](image/Oct17/1792283075738.png)

![1792283096309](image/Oct17/1792283096309.png)

![1792283103994](image/Oct17/1792283103994.png)

![1792283026654](image/Oct17/1792283026654.png)

![1792283206483](image/Oct17/1792283206483.png)

![1792283217532](image/Oct17/1792283217532.png)

![1792283150954](image/Oct17/1792283150954.png)

![1792283160089](image/Oct17/1792283160089.png)

![1792283167824](image/Oct17/1792283167824.png)
