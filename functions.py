#### IMPORTS ####
import numpy as np
from scipy import stats

#### FUNCTIONS ####
# Std. gaussian pdf and cdf
norm = stats.norm(0,1)

def helper(t1:float, t2:float,b1:float,b2:float,sigma:float,r:float):
    return norm.cdf(1/(sigma*np.sqrt(t2-t1))*(np.log(b2/b1)-(r-sigma**2/2)*(t2-t1)))

def helper_mesh(i,j,trng,b_res,sigma,r):
    t1 = trng[i]
    b1 = b_res[i]
    t2 = trng[j]
    b2 = b_res[j]
    return np.exp(-r*(t2-t1))*norm.cdf(1/(sigma*np.sqrt(t2-t1))*(np.log(b2/b1)-(r-sigma**2/2)*(t2-t1)))

def helper_vec(t1:float, t2:np.ndarray,b1:float,b2:np.ndarray,sigma:float,r:float):
    res = np.zeros_like(t2)
    res[:-1] = np.exp(-r*(t2[:-1]-t1))*helper(t1,t2[:-1],b1,b2[:-1],sigma,r)
    return res
        
def g(t:float,bt:float,sigma:float,r:float,T:float,K:float):
    dp = 1/(sigma*np.sqrt(T-t))*(np.log(bt/K)+(r+sigma**2/2)*(T-t))
    put = K*np.exp(-r*(T-t))*norm.cdf(-(dp-sigma*np.sqrt(T-t)))-bt*norm.cdf(-dp)
    return put

def calc_step(bt: float, i:int, trng:np.ndarray, b_res:np.ndarray,sigma:float, r:float,K:float,T:float,n:int,fp = False):

    t=trng[i]
    
    # Trapezoidal 
    Gs = np.zeros_like(b_res[:i+1])

    Gs = r*K*helper_vec(t,trng[:i+1],bt,b_res[:i+1],sigma,r)

    Gs[0]=Gs[0]/2
    Gs[-1]=(r*K/2)/2
    
    int_G = np.sum(Gs)*T/n

    # Return whether fixed point or backward
    if fp==1:
        return K-int_G-g(t,bt,sigma,r,T,K)
    if fp==2:
        dp = 1/(sigma*np.sqrt(T-t))*(np.log(bt/K)+(r+sigma**2/2)*(T-t))
        return (-int_G+K*(1-np.exp(-r*(T-t))*norm.cdf(-dp+sigma*np.sqrt(T-t))))/norm.cdf(dp)
    
    return bt-K+int_G+g(t,bt,sigma,r,T,K)


