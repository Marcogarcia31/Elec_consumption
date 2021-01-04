from math import comb

### Function that gives the minimal n_iter parameter value 
### to reach an optimal combination in a Randomized grid, given that the proportion 
### optimal combinations is equal to 0.05, with a level of certainty of 0.95
def min_n_iter(N, proba_success):
    
    ### number of iterations
    n_iter = 0
    
    ### proba of reaching success ie getting at least one winning ball for given n
    proba = 0
    
    n_winning_comb = int(proba_success*N)
    n_losing_comb = int((1-proba_success)*N)+1
    while proba < 0.95:
        
        ### Increments n to reach proba>= 0.95
        n += 1
        
        ### Reinitializes proba for new n_iter value
        proba = 0
        
        for i in range(max(0, n_iter - n_losing_comb),min(n_winning_comb, n_iter)-1):
            proba = proba + comb(n_winning_comb, i+1)*comb(n_losing_comb, n_iter - (i+1))/comb(N, n_iter)
    
    
    return n_iter


n_iter = min_n_iter(100, 0.05)

print(n_iter)
    