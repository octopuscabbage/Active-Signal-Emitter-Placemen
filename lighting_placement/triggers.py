import math
import numpy as np
from scipy.stats import norm

def every_n_trigger(n,current_budget):
    return (current_budget % n) == 0

def compute_logprob(observations,mean,std):
    # ll = 0
    # scipy_ll = 0
    # for cur_mean,cur_std, cur_obs in zip(mean,std,observations):
    #     ll += norm(cur_mean,cur_std).logpdf(cur_obs)
    #     p_cur = norm.pdf(cur_obs,cur_mean,cur_std)
    #     if p_cur != 0:
    #         scipy_ll += np.log(p_cur)
    #assert ll == np.sum(norm(mean,std).logpdf(observations))
    ll_so = -np.sum(np.log(2*math.pi*(std**2))/2 + ((observations-mean)**2)/(2 * (std**2)))
    return ll_so

def logprob_percent_trigger(percent,previous_logprob,current_logprob):
    percent_change = (current_logprob - previous_logprob) / np.abs(previous_logprob) * 100
    #Since it's percent decrease we want less than the negative percent
    return percent_change < -percent

def logprob_fraction_trigger(fraction,previous_logprob,current_logprob):
    #percent_change = (current_logprob - previous_logprob) / np.abs(previous_logprob) * 100
    #cur_fraction = np.abs(previous_logprob) / np.abs(current_logprob)
    fractional_amount = previous_logprob * fraction
    #Since it's percent decrease we want less than the negative percent
    return  current_logprob < fractional_amount
