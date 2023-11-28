#### IMPORTS ####
import numpy as np
from scipy.optimize import newton
from functions import *

import multiprocessing
from itertools import repeat

import time



def backward_recursion(adaptive_offset:bool,trng:np.ndarray, b_res:np.ndarray,sigma:float, r:float,K:float,T:float,n:int):
    par_offset=0
    for i in range(n):
        if adaptive_offset and i > 0:
            par_offset = (b_res[i]-b_res[i-1])/1.1 # Seems to work nicely
        b_res[i+1]=newton(calc_step,b_res[i]+par_offset,args = (i+1,trng,b_res,sigma,r,K,T,n),maxiter=100)
    
    return b_res


def picard_iteration(type:int, b_res_mat:np.ndarray,numiter:int, trng:np.ndarray, sigma:float, r:float,K:float,T:float,n:int):
    for j in range(numiter-1):
        for i in range(n):
            b_res_mat[i+1,j+1]=calc_step(b_res_mat[i+1,j],i+1,trng,b_res_mat[:,j],sigma,r,K,T,n,fp=type)
    return b_res_mat

def picard_iteration_pythonic(b_res_mat:np.ndarray,numiter:int, trng:np.ndarray, sigma:float, r:float,K:float,T:float,n:int):
    """
    A more pythonic implementation, avoiding the inner loop. 
    However no speedup compared to `picard_iteration` 
    """
    for j in range(numiter-1):
        b_res_mat[1:,j+1]=picard_step(b_res_mat[:,j],trng,K,r,sigma,T,n)
            
    return b_res_mat

def picard_iteration_parallel(type:int, b_res_mat:np.ndarray,numiter:int, trng:np.ndarray, sigma:float, r:float,K:float,T:float,n:int):
    pool = multiprocessing.Pool()
    
    for j in range(numiter-1):     
        args = zip(b_res_mat[1:,j],np.arange(n)+1,repeat(trng),repeat(b_res_mat[:,j]),repeat(sigma),repeat(r),repeat(K),repeat(T),repeat(n),repeat(type))
        b_res_mat[1:,j+1] = pool.starmap(calc_step,args)        
    return b_res_mat

def picard_iteration_parallel_safe(type:int, b_res_mat:np.ndarray,numiter:int, trng:np.ndarray, sigma:float, r:float,K:float,T:float,n:int):
    """A safer version"""    
    for j in range(numiter-1):     
        args = zip(b_res_mat[1:,j],np.arange(n)+1,repeat(trng),repeat(b_res_mat[:,j]),repeat(sigma),repeat(r),repeat(K),repeat(T),repeat(n),repeat(type))
        with multiprocessing.Pool() as pool:
            b_res_mat[1:,j+1] = pool.starmap(calc_step,args)        
    return b_res_mat

def calc_price(x,b_res,trng,sigma,r,K,T,n):
    return calc_step(x,n,trng,b_res,sigma,r,K,T,n)+K-x


if __name__=="__main__":
    #### MODEL PARAMETERS ####
    x0 = 44
    K = 40
    r = 0.06
    sigma = 0.2
    T = 1

    #### GRID SETUP ####
    n = 400
    trng = np.arange(0,T+T/n,T/n)[::-1]

    #### Choose mthd ####
    use_bw = False

    #### Recursion setup ####
    adaptive_offset = True
    b_res = np.ones_like(trng)*K

    #### Picard setup ####
    type = 2
    numiter = 4
    smart_start = True
    if smart_start:
        temp = K/(1+sigma**2/(2*r))
        b_res_mat = np.tile(temp+(K-temp)*np.exp(-np.sqrt(0.5*(trng[::-1])**(0.85))),(numiter,1)).T
    else:
        b_res_mat = np.ones((n+1,numiter))*K

    

    # Backward Recursion
    start = time.perf_counter()
    res_bw = backward_recursion(adaptive_offset,trng,b_res,sigma,r,K,T,n)
    end = time.perf_counter()
    print("BW = {}ms".format((end - start)*1000))
    
    # Picard iteration
    start = time.perf_counter()
    res_p = picard_iteration(type,b_res_mat, numiter,trng,sigma,r,K,T,n)
    end = time.perf_counter()
    print("picard = {}ms".format((end - start)*1000))   

    # Pythonic way of picard(has division errors that we ignore)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        res_python = picard_iteration_pythonic(b_res_mat, numiter,trng,sigma,r,K,T,n)
        end = time.perf_counter()
        print("picard(pythonic) = {}ms".format((end - start)*1000))   

    assert  (abs(res_p[:,-1]-res_python[:,-1])<1e-7).all()

    
    # Picard iteration with parallel
    start = time.perf_counter()
    res_par = picard_iteration_parallel(type,b_res_mat, numiter,trng,sigma,r,K,T,n)
    end = time.perf_counter()
    print("picard(Parallel) = {}ms".format((end - start)*1000))   

    # Picard iteration with parallel (safe)
    start = time.perf_counter()
    res_par = picard_iteration_parallel_safe(type,b_res_mat, numiter,trng,sigma,r,K,T,n)
    end = time.perf_counter()
    print("picard(Parallel(safe)) = {}ms".format((end - start)*1000))   

    # Calulate the price
    print("\nprice: {}".format(calc_price(x0,res_par[:,-1],trng,sigma,r,K,T,n)))

    assert  (abs(res_bw-res_p[:,-1])<0.1).all()
    assert  (abs(res_par[:,-1]-res_p[:,-1])<1e-7).all()


 