
# coding: utf-8

# # Statistics topics in data science: hypothesis testing & Bayes inference
# 
# ## Hypothesis Testing

# A hypothesis test is used to determine whether observed data deviates from an expected result.

# A simple example: let's suppose we expect the age of women in San Francisco to be normally distributed with mean 37 and standard deviation 11.1. We take a survey of 100 women in SF and calculate their average age. What have we learned from this observation?

# * Protocol:
#     1. You have a null hypothesis, $H_0$; (the mean age of women in San Francisco is 37) and an alternative hypothesis (mean is > 37). 
#     2. You choose a significance level at which you will accept or reject the null hypothesis
#     3. You do a statistical test based on some sample data and calculate a $p$ value
#     4. You decide whether to accept or reject $H_0$

#   
#  
#  
# * The statistical test in step 3 involves calculation of a z or t-score.  
# 
# 
# #### z-score
# 
# $$z = \frac{x-\mu}{\frac{\sigma}{\sqrt n}}$$
# 
# where $x$ is the sample mean, $\mu$ is the population mean, and $\sigma$ is the *population* standard deviation, and $n$ is the sample size. This is a measure of how many standard deviations away from the true/expected mean is the sample mean. 
# 
# If we are just calculating the z score based on 1 observation, then in the above formula $n$ = 1, which is why we often see z written without the $\sqrt n$ in the denominator. Confusing!  
# 
# #### t-score
# Small sample, or we don't know population standard deviation 
# 
# $$t = \frac{x-\mu}{\frac{s}{\sqrt n}}$$
# 
# where here, $s$ is the *sample* standard deviation. The quantity $\frac{s}{\sqrt n}$ is also known as standard error. If we know the population standard deviation $\sigma$ then we can substitute it for $s$ in the calculation of t-score. The Student T distribution is like a Gaussian normal distribution but with wider tails. At large $n$ the student T approaches the normal distribution. 

# In[ ]:

# Since we know population standard deviation ~and~ it's a large-ish sample size,
# we can use the standard normal distribution to perform our hypothesis test
import math 
from scipy import stats

z = (38.9-37)/(11.1/math.sqrt(100))
print "z = {}".format(str(round(z,2)))
print "p = {}".format(str(1 - round(stats.norm.cdf(z),3)))


# At a 95% confidence level, we do a one-tailed test because the alternative hypothesis was that the mean age is greater than 37, so since p < 0.05, we would reject the null hypothesis.

# #### Danger of relying on $p$ values
# 
# * a low $p$ value doesn't mean that the null hypothesis is false. It may mean that the null hypothesis is true and that an improbable event occured.
# * people misrepresent $p$ value results by creating a hypothesis after the $p$ value is calculated 
# * people may sample until $p$ is "statistically significant" (< 0.05)
# * see [this article](http://www.nature.com/news/scientific-method-statistical-errors-1.14700)   
#     
#     
# * Ways to combat
#     * always report confidence interval & sample size along with p value
#     * consider a Bayesian approach

# 
# <img src="my_confusion_matrix.png" alt="" style="width: 500px;"/>
# 
# If you test at a high confidence level then you're more likely to be correct when you reject the null hypothesis (less type I errors) but more likely to inappropriately accept the null (more type II errors) so you increase precision but decrease recall.

# ## Bayes Theorem

# Bayes Rule is used to estimate probability of an event given some prior knowledge. 

# * Prior probability + test evidence —> posterior probability 
# 
# 
# * Bayes Rule formula
# $$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$
# 
# Common example: if we know that the probability of having a certain type of cancer is 1% ($P(A)$), and that the probability of having a positive test result (B) if person has cancer is 90% ($P(B | A)$), what is the probability that the person has cancer if their test is positive?
# 
# #### Some terminology
# $P(A)$ is known as the *prior* : what we know about the probability of having cancer before we do the test.  
# $P(B | A)$ is the *likelihood*: the likelihood of getting a positive result if a person has cancer based on the test  
# $P(B)$ is the overall probability of having a positive test result, and is used normalization factor, also known as test *evidence*   
# $P(A | B)$ is the result, or *posterior* probability of having cancer, that we've calculated based on prior knowledge
# 
# $P(B)$, the evidence, requires some other information: say the probability of having a **negative** test result and **not** having cancer is also 90%. That means that 10% of healthy people will get a positive result also. So P(B) = true positives + false positives = (0.9)(0.01) + (0.1)(0.99) = 0.09 + 0.099 = 0.108

# In[ ]:

prior = 0.01
likelihood = 0.9
norm = 0.108
posterior = prior * likelihood / norm
print posterior


# ### Conjugate priors & posterior sampling
# 
# Another way of looking at Bayes inference: suppose we are trying to measure $P(\theta | X)$ where $\theta$ includes model parameters like mean, and X is again the test evidence. So we are calculating the probability of our model parameters, given our test observations. 
# 
# For this analysis, we'll go back to our example of the age of women in San Francisco, and I want to know the mean of the population based on some test data X. So we want to calculate $P(\mu | X)$.
# 
# Let's assume that based on some *prior* historical data, we believe the mean $\mu$ follows the normal distribution centered around 37. That's our $P(\mu)$. Our *likelihood* of the test data, given this distribution for $\mu$, can then be calculated fairly easily. 
# 
# But what about the denominator $P(X)$? Like in the simple cancer example, it's not trivial to calculate; it's the overall likelihood of obtaining the test data X over all possible model parameters (means). This makes it difficult to analytically solve for the posterior probability distribution.
# 
# But because we chose the prior to be a normal distribution, we CAN solve for the parameters of the posterior distribution exactly; it is also a normal distribution. That is because of the concept of **conjugate priors**: the normal distribution is conjugate to itself with respect to a Gaussian likelihood function. A mathematical proof and more information on this can be found [here](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxiYXllc2VjdHxneDplNGY0MDljNDA5MGYxYTM). There are other families of conjugates, a diagram of which can be found at the end of the notebook.

# In[ ]:

import numpy as np


# In[ ]:

# Parameters of the posterior distribution: normal because of the conjugate prior chosen
def mu_post(sample_data, mu_0, sigma_0):
    sigma_sample = np.std(sample_data)
    return (mu_0/sigma_0**2 + sum(sample_data)/sigma_sample**2) / (1.0/sigma_0**2 + len(sample_data)/sigma_sample**2)

def sigma_post(sample_data, sigma_0):
    return math.sqrt(1.0 / (1.0/sigma_0**2 + len(sample_data)/np.std(sample_data)**2))


# In[ ]:

# generate some test data
sample_data = np.random.normal(38.0, 12.0, 100)

# calculate parameters of posterior distribution based on a mean that is normally distributed with mean 37.0 and std 2
print mu_post(sample_data, 37.0, 2.0)
print sigma_post(sample_data, 2.0)


# This is an ideal situation, in reality you probably don't have conjugate priors and you don't know anything about your posterior distribution. You can deal with this using **posterior sampling**. See this [blog post](http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/)
# 
# Markov Chain Monte Carlo (MCMC) is a set of methods for doing posterior sampling. Basically, it involves making an initial guess on your posterior parameters, and moving around in the parameter space to parameters which explain your data better than the previous guess (by calculating prior * likelihood using new parameters). We choose whether or not to accept the new parameter and this becomes a sample of our posterior distribution.

# ## Putting it together with hypothesis testing

# Hypothesis testing falls under a frequentist perspective - probability is related to measured frequency of events. 
# 
# 
# Bayesian probability is related to one's knowledge about events.

# In hypothesis testing, we test the likelihood of seeing test data given model parameters like mean and standard deviation. In Bayesian inference, we consider the **DATA** to be fixed and the model parameters to be variable.
# 
# 
# It's the difference between calculating $P(x | \theta)$ vs. $P(\theta | x)$, where $x$ are your test data and $\theta$ are your model parameters.     
# 
# 
# 
# 
# 

# In[ ]:












# 
# 
# 
# 
# 
# 
# 
# 
# ![](conjugate_prior_diagram.png)
# 
# (source: https://www.johndcook.com/blog/conjugate_prior_diagram/)

# In[ ]:




# In[ ]:




# In[ ]:



