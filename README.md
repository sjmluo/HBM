# Higher-Order Boltzmann Machine
An implementation of the Higher-Order Boltzmann Machine. The implementation represents the boltzmann machine using the information geometry formulation of the log-linear model. The model is trained by minimising the KL-divergence and then using a combination of Gibbs sampling and Annuealed Importance Sampling (AIS) to estimate the partition function.

Please see the following paper for more details:
* Luo, S., Sugiyama, M.: ** Bias Variance Trade-Off in Hierarchical Probabilistic Models using Higher-Order Feature Interactions, AAAI 2019 **

## Usage
An example on how to run the code:
```
import numpy as np

X = [[1,0,0,0],
     [0,0,0,1],
     [0,0,1,1],
     [1,1,1,1],
     [1,0,0,0],
     [0,0,0,1],
     [0,0,1,1],
     [1,1,1,1],
     [0,1,1,0],
     [0,1,1,1],
     [0,1,1,1],
     [0,0,1,0],
     [0,0,1,1],
     [0,1,0,0],
     [0,1,0,1],
     [1,0,1,1],
     [1,1,1,1]]
X = np.array(X)
HBM = Higher_Order_Boltzmann_Machine(X,order=2)
HBM.train(lr=1, N_gibbs_samples=10000, burn_in=3, MAX_ITERATIONS=100, verbose=True)
predict_C = [[0,1,0,1],
             [1,0,1,1]]
predict_C = np.array(predict_C)
p_vec, p_lower_vec, p_upper_vec = HBM.ais2p(predict_C)
```