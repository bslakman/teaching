
# coding: utf-8

# ## Hypothesis Testing

# A hypothesis test is used to determine whether observed data deviates from an expected result.

# * Protocol:
#     * You have a null hypothesis, $H_0$ (the mean age of women in Boston is 27) and an alternative hypothesis, $^H^$ (mean is > 27). 
#     * You choose a confidence level at which you will accept or reject the null hypothesis, $\alpha$
#     * You do a statistical test based on some sample data to calculate a $p$ value
#     * You decide whether to accept or reject $H_0$ (and you're either right or wrong!).
# * Danger of $p$ values
#     * a low $p$ value doesn't mean that the null hypothesis is false. It may just mean that the null hypothesis is true and that an improbable event occured!
#     * people misrepresent $p$ value results by creating a hypothesis after the $p$ value is calculated!
# * z-score
# * t-test: small sample, or we don't know population standard deviation 
#     * one sample
#     * two sample
#     * paired
# * have 4 way window with true/false pos/neg, whether this means we accept/reject null hypothesis, relationship to precision/recall
# 
# 
# ![caption](confusion_matrix.png)

# ## Bayes Theorem

# Bayes Rule is used to estimate probability of an event given some prior knowledge. 

# * Prior probability + test evidence —> posterior probability 
# * Bayes Rule formula
# $$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$
# * relationship to hypothesis testing

# In[ ]:



