# This file contains a routine for private mean estimation
# Source: https://github.com/twistedcubic/coin-press
# Publication: https://arxiv.org/abs/2006.06618

'''
Privately estimating covariance.
'''

import torch
import numpy as np
import math

def cov_est_step(X, A, rho, cur_iter, args):
    """
    One step of multivariate covariance estimation, scale cov a.
    """
    W = torch.mm(X, A)
    n = args.n
    d = args.d

    #Hyperparameters
    gamma = gaussian_tailbound(d, 0.1)
    eta = 0.5*(2*(np.sqrt(d/n)) + (np.sqrt(d/n))**2)
    
    #truncate points
    W_norm = np.sqrt((W**2).sum(-1, keepdim=True))
    norm_ratio = gamma / W_norm
    large_norm_mask = (norm_ratio < 1).squeeze()
    
    W[large_norm_mask] = W[large_norm_mask] * norm_ratio[large_norm_mask]
    #noise
    Y = torch.randn(d, d)
    noise_var = (gamma**4/(rho*n**2))
    Y *= np.sqrt(noise_var)    
    #can also do Y = torch.triu(Y, diagonal=1) + torch.triu(Y).t()
    Y = torch.triu(Y)
    Y = Y + Y.t() - Y.diagonal().diag_embed() #Don't duplicate diagonal entries
    Z = (torch.mm(W.t(), W))/n
    #add noise    
    Z = Z + Y
    #ensure psd of Z
    Z = psd_proj_symm(Z)
    
    U = Z + eta*torch.eye(d)
    inv = torch.inverse(U)
    invU, invD, invV = inv.svd()
    inv_sqrt = torch.mm(invU, torch.mm(invD.sqrt().diag_embed(), invV.t()))
    A = torch.mm(inv_sqrt, A)
    return A, Z

def cov_est(X, args ):
    """
    Multivariate covariance estimation.
    Returns: zCDP estimate of cov.
    """
    A = torch.eye(args.d) / np.sqrt(args.u)
    assert len(args.rho) == args.t
    
    for i in range(args.t-1):
        A, Z = cov_est_step(X, A, args.rho[i], i, args)
    A_t, Z_t = cov_est_step(X, A, args.rho[-1], args.t-1, args)
    
    cov = torch.mm(torch.mm(A.inverse(), Z_t), A.inverse())
    return cov

def gaussian_tailbound(d,b):
    return ( d + 2*( d * math.log(1/b) )**0.5 + 2*math.log(1/b) )**0.5

def mahalanobis_dist(M, Sigma):
    Sigma_inv = torch.inverse(Sigma)
    U_inv, D_inv, V_inv = Sigma_inv.svd()
    Sigma_inv_sqrt = torch.mm(U_inv, torch.mm(D_inv.sqrt().diag_embed(), V_inv.t()))
    M_normalized = torch.mm(Sigma_inv_sqrt, torch.mm(M, Sigma_inv_sqrt))
    return torch.norm(M_normalized - torch.eye(M.size()[0]), 'fro')

''' 
Functions for mean estimation
'''

##    X = dataset
##    c,r = prior knowledge that mean is in B2(c,r)
##    t = number of iterations
##    Ps = 
def multivariate_mean_iterative(X, c, r, t, Ps):
    for i in range(t-1):
        c, r = multivariate_mean_step(X, c, r, Ps[i])
    c, r = multivariate_mean_step(X, c, r, Ps[t-1])
    return c

def multivariate_mean_step(X, c, r, p):
    n, d = X.shape

    ## Determine a good clipping threshold
    gamma = gaussian_tailbound(d,0.01)
    clip_thresh = min((r**2 + 2*r*3 + gamma**2)**0.5,r + gamma) #3 in place of sqrt(log(2/beta))
        
    ## Round each of X1,...,Xn to the nearest point in the ball B2(c,clip_thresh)
    x = X - c
    mag_x = np.linalg.norm(x, axis=1)
    outside_ball = (mag_x > clip_thresh)
    x_hat = (x.T / mag_x).T
    X[outside_ball] = c + (x_hat[outside_ball] * clip_thresh)
    
    ## Compute sensitivity
    delta = 2*clip_thresh/float(n)
    sd = delta/(2*p)**0.5
    
    ## Add noise calibrated to sensitivity
    Y = np.random.normal(0, sd, size=d)
    c = np.sum(X, axis=0)/float(n) + Y
    r = ( 1/float(n) + sd**2 )**0.5 * gaussian_tailbound(d,0.01)
    return c, r

def L1(est): # assuming 0 vector is gt
    return np.sum(np.abs(est))
    
def L2(est): # assuming 0 vector is gt
    return np.linalg.norm(est)

def cov_nocenter(X):
    X = X
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

def cov(X):
    X = X - X.mean(0)
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

'''
PSD projection
'''
def psd_proj_symm(S):
    U, D, V_t = torch.svd(S)
    D = torch.clamp(D, min=0, max=None).diag()
    A = torch.mm(torch.mm(U, D), U.t()) 
    return A

'''
Mean Estimation Methods --------------------------------------------------------
'''

'''
Fine mean estimation algorithm 
 - list params are purely for graphing purposes and can be ignored if not needed
returns: fine DP estimate for mean
'''
def fineMeanEst(x, sigma, R, epsilon, epsilons=[], sensList=[], rounding_outliers=False):
    B = R+sigma*3
    sens = 2*B/(len(x)*epsilon) 
    epsilons.append([epsilon])
    sensList.append([sens])
    if rounding_outliers:
        for i in x:
            if i > B:
                i = B
            elif i < -1*B:
                i =  -1*B
    noise = np.random.laplace(loc=0.0, scale=sens)
    result = sum(x)/len(x) + noise 
    return result

'''
Coarse mean estimation algorithm with Private Histogram
returns: [start of intrvl, end of intrvl, freq or probability], bin number
- the coarse mean estimation would just be the midpoint of the intrvl (in case this is needed)
'''
def privateRangeEst(x, epsilon, delta, alpha, R, sd):
    # note alpha ∈ (0, 1/2)
    r = int(math.ceil(R/sd))
    bins = {}
    for i in range(-1*r,r+1):
        start = (i - 0.5)*sd # each bin is s ((j − 0.5)σ,(j + 0.5)σ]
        end = (i + 0.5)*sd 
        bins[i] = [start, end, 0] # first 2 elements specify intrvl, third element is freq
    # note: epsilon, delta ∈ (0, 1/n) based on https://arxiv.org/pdf/1711.03908.pdf Lemma 2.3
    # note n = len(x)
    # set delta here
    L = privateHistLearner(x, bins, epsilon, delta, r, sd)
    return bins[L], L


# helper function
# returns: max probability bin number
def privateHistLearner(x, bins, epsilon, delta, r, sd): # r, sd added to transmit info
    # fill bins
    max_prob = 0
    max_r = 0

    # creating probability bins
    for i in x:
        r_temp = int(round(i/sd))
        if r_temp in bins:
            bins[r_temp][2] += 1/len(x)
        
    for r_temp in bins:
        noise = np.random.laplace(loc=0.0, scale=2/(epsilon*len(x)))
        if delta == 0 or r_temp < 2/delta:
            # epsilon DP case
            bins[r_temp][2] += noise
        else:
            # epsilon-delta DP case
            if bins[r_temp][2] > 0:
                bins[r_temp][2] += noise
                t = 2*math.log(2/delta)/(epsilon*len(x)) + (1/len(x))
                if bins[r_temp][2] < t:
                    bins[r_temp][2] = 0
        
        if bins[r_temp][2] > max_prob:
            max_prob = bins[r_temp][2]
            max_r = r_temp
    return max_r


'''
Two shot algorithm
- may want to optimize distribution ratio between fine & coarse estimation
eps1 = epsilon for private histogram algorithm
eps2 = epsilon for fine mean estimation algorithm
returns: DP estimate for mean
'''
def twoShot(x, eps1, eps2, delta, R, sd):
    alpha = 0.5
    # coarse estimation
    [start, end, prob], r = privateRangeEst(x, eps1, delta, alpha, R, sd)
    for i in range(len(x)):
        if x[i] < start - 3*sd:
            x[i] = start - 3*sd
        elif x[i] > end + 3*sd:
            x[i] = end + 3*sd
    # fine estimation with smaller range (less sensitivity)
    est = fineMeanEst(x, sd, 3.5*sd, eps2)
    return est