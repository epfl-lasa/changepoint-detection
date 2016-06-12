# changepoint-detection

###Project Goal

Implement an Online Changepoint Detection Algorithm for Segmenting Kinesthetic Data from Human Demonstrations or in Human-Robot interaction scenarios, such a problem is known in Machine Learning and Statistics Literature as "Bayesian Changepoint Detection".

####Bayesian Changepoint Detection:

Methods to get the probability of a changepoint in a time series. Both online and offline methods are availeble. Read the following papers to really understand the methods:

[1] Ryan P. Adams, David J.C. MacKay, Bayesian Online Changepoint Detection, arXiv 0710.3742 (2007)

[2a] Paul Fearnhead, Exact and Efficient Bayesian Inference for Multiple Changepoint problems, Statistics and computing 16.2 (2006), pp. 203--213

[2b] P Fearnhead and Z Liu. Online inference for multiple changepoint problems. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(4):589-605, 2007.

[3] Xuan Xiang, Kevin Murphy, Modeling Changing Dependency Structure in Multivariate Time Series, ICML (2007), pp. 1055--1062

[4] Scott Niekum et al. CHAMP: Changepoint Detection Using Approximate Model Parameters.
(This is an approximation to [2b])

###Implementations:
- A matlab implementation of [1] is found in ./matlab-implementation directory. 

- A python implementation of [1,2a,3] is found in ./python-implementation directory.
#####Tips on python implementation:
  - The online version is basically a translation of the matlab version of the paper from
  author [1], found in ./matlab-implementation.
  - The offline version is an implementation based on [2a] and [3].
  - A conversation I found online that might be interesting:
  
    Q: Is it possible to make Ryan Adams algorithm [1] to work on multivariate data too?

    A: The change should be relatively easy, but a bit time consuming. It's only updating the student t distribution to handle multivariate data correctly. 

- A c++ implementation of [4] is found in (http://wiki.ros.org/changepoint) and
(https://github.com/sniekum/changepoint) 


