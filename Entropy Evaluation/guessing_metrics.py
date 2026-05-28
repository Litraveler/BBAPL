import numpy as np

def min_entropy(most_probable_event):
    '''
    Hinf
    @param most_probable_event: compute the most probable event
    '''
    return -np.log2(most_probable_event)

def hartley_entropy(prob_list):
    return np.log2(len(prob_list))

def shannon_entropy(prob_list):
    return np.sum(np.multiply(prob_list, -np.log2(prob_list)))

def guesswork(prob_list):
    ''' Expected number of guesses to find X when attacker proceeds in optimal order.
    @param: prob_list: the probability of a specific sample matching a specific user X
    @return the sum of prob*nummer of guesses
    '''
    return np.sum(np.multiply(prob_list, np.arange(1,len(prob_list)+1)))

def beta_success_rate(beta, prob_list):
    '''
    Expected amount of successes for an attacker limited to beta guesses
    @param: beta, the amount of guesses
    @param: prob_list, the probability of a specific sample being matched with a specific user
    @constraint: beta < len(prob_list)
    @return b_success_rate, defined as the sum of the top of the problist
    '''
    return np.sum(prob_list[:beta])

def beta_entropy(beta, prob_list):
    '''
    Beta entropy is the effective key length of beta_success_rate.
    It is computed by looking for the size of the equivalent uniform distribution
    Equivalent impying uniform distribution that has the same beta_success_rate.
    '''
    return np.log2(beta/beta_success_rate(beta, prob_list))


def alfa_work_factor(alfa, prob_list):
    '''
    Number of guesses per account needed to break at least a fraction alfa of accounts
    @param alfa: desired fraction of accounts to break, alfa in [0,1]
    @prob_list: probability of a specific sample being matched with a specific user
    '''
    return np.searchsorted(np.cumsum(prob_list),alfa)+1

def alfa_work_factor_entropy(alfa, prob_list):
    '''
    The effective key length of alfa_work_factor.
    Computed by looking for the sive of the equivalent uniform distribution.
    '''
    alfa_wf = alfa_work_factor(alfa,prob_list)
    return np.log2(alfa_wf/beta_success_rate(alfa_wf, prob_list))

def alfa_guess_work(alfa, prob_list):
    '''
    Expected number of guesses per account to achieve a success rate alfa
    @param alfa: success rate
    @param prob_list: probability of a specific sample being matched with a specific user
    '''
    alfa_wf = alfa_work_factor(alfa, prob_list)
    beta_sr = beta_success_rate(alfa_wf, prob_list)
    return (1-beta_sr)*alfa_wf + guesswork(prob_list[:alfa_wf])

def alfa_guess_work_entropy(alfa, prob_list):
    '''
    Effective key length of alfa_guess_work
    Attacker will have a success every alfa_guess_work/alfa guesses.
    Uniform distribution attacker breaks an account every (N+1)/2 guesses.
    '''
    alfa_gw = alfa_guess_work(alfa, prob_list)
    print("alfa_gw: " + str(alfa_gw))
    alfa_wf = alfa_work_factor(alfa, prob_list)
    print("alfa_wf: " + str(alfa_wf))
    beta_sr = beta_success_rate(alfa_wf, prob_list)
    print("beta_sr: " + str(beta_sr))
    return np.log2(2*alfa_gw/beta_sr-1)+np.log2(1/(2-beta_sr))
