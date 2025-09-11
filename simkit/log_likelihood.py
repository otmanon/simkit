import numpy as np
import scipy as sp

# def log_likelihood(u : np.ndarray,  mu : np.ndarray=None, cov : Covariance =None):
    
#     n = u.shape[0]
#     if u.ndim == 1:
#         u = u.reshape(-1, 1)

#     if mu is None:
#         mu = np.zeros((u.shape[0], 1))
#     mu = mu.reshape(-1, 1)

#     d = u - mu
#     ll = -0.5 * d.T @ cov.Ci() @ d  + 0.5 * np.log(cov.Ci().diagonal()).sum() #- n * 0.5 * np.log(2 * np.pi)
    
#     return ll


def log_likelihoods_diagonal(U : np.ndarray, covs, covs_inv=None, sumlogcovs=None,  mus : np.ndarray=None, marginalize=None, marginalize_thresh=None):
    n = U.shape[0]
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    lls = np.zeros((len(covs), 1))
    for i in range(len(covs)):
        if mus is None:
            d = U
        else:   
            mu = mus[:, i].reshape(-1, 1)
            d = U - mu
        
        cov = covs[i]
        if covs_inv is None:
            Ci = sp.sparse.diags(1.0/cov.diagonal()).tocsc()
        else:
            Ci = covs_inv[i]
        
        if marginalize_thresh is not None:
            inds = np.where(cov.diagonal() > marginalize_thresh)[0]
            d = d[inds, :]
            Ci = Ci[inds, :][:, inds]

        if marginalize is not None:
            if isinstance(marginalize, list):
                d = d[marginalize[i], :]
                Ci = Ci[marginalize[i], :][:, marginalize[i]]
                cov = cov.tocsc()[marginalize[i], :][:, marginalize[i]]
            else:
                d = d[marginalize, :]
                Ci = Ci[marginalize, :][:, marginalize]
                cov = cov.tocsc()[marginalize, :][:, marginalize]   

        if sumlogcovs is None:
            sumlogcov = np.sum(np.log(cov.diagonal()))
        else:
            sumlogcov = sumlogcovs[i]
        
        nn = d.shape[0]
        ll = -0.5 * np.sum(d.T @ (Ci @ d)) - 0.5 * sumlogcov -0.5* np.log(2 * np.pi) * nn

        lls[i] = ll
    return lls

def conditional_likelihoods_diagonal(U : np.ndarray, covs, covs_inv=None, sumlogcovs=None,  mus : np.ndarray=None, marginalize=None, marginalize_thresh=None):
    log_likelihoods = log_likelihoods_diagonal(U, covs, covs_inv=covs_inv, sumlogcovs=sumlogcovs, mus=mus, marginalize=marginalize,  marginalize_thresh=marginalize_thresh)
    log_denominator = sp.special.logsumexp(log_likelihoods)
    conditional_ll = log_likelihoods - log_denominator
    conditional_l = np.exp(conditional_ll)
    return conditional_l, conditional_ll

def log_likelihoods_cov_inv(U : np.ndarray, covs_inv,  mus : np.ndarray=None, marginalize=None):
    n = U.shape[0]
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    if mus is None:
        mus = np.zeros((U.shape[0], len(covs_inv)))

    lls = np.zeros((len(covs_inv), 1))
    for i in range(len(covs_inv)):
        mu = mus[:, i].reshape(-1, 1)
        d = U - mu
        Ci =  covs_inv[i]

        if marginalize is not None:
            if len(marginalize) == len(covs_inv):
                d = d[marginalize[i], :]
            
            else:
                d = d[marginalize, :]
            
        nn = d.shape[0]
        ll = -0.5 * np.sum(d.T @ Ci @ d) - 0.5 * np.sum(np.log(((Ci.diagonal())))) -0.5* np.log(2 * np.pi) * nn

        lls[i] = ll
    return lls