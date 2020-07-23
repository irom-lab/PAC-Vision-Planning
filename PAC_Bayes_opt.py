import cvxpy as cvx
import scipy
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

def optimize_PAC_bound(costs_precomputed, p0, delta):
    '''Optimize McAllester (Maurer) PAC bound using Relative Entropy Programming'''    
    # Number of actions
    L = len(p0)
    
    # Number of environments
    m = np.shape(costs_precomputed)[0]
    
    # Discretize lambdas
    lambdas = np.linspace(0,1,100)
    
    # Initialize vectors for storing optimal solutions
    taus = np.zeros(len(lambdas))
    ps = len(lambdas)*[p0]

    for k in range(len(lambdas)):

        lambda0 = lambdas[k]

        # Create cost function variable
        tau = cvx.Variable()

        # Create variable for probability vector
        p = cvx.Variable(L)

        cost_empirical = (1/m)*cvx.sum(costs_precomputed*p)

        # Constraints
        constraints = [lambda0**2 >= (cvx.sum(cvx.kl_div(p, p0)) + np.log(2*np.sqrt(m)/delta))/(2*m), lambda0 == (tau - cost_empirical), p >= 0, cvx.sum(p) == 1]
        
        prob = cvx.Problem(cvx.Minimize(tau), constraints)

        # Solve problem
        opt = prob.solve(verbose=False, solver=cvx.MOSEK) # , max_iters=3000)

        # Store optimal value and optimizer
        if (opt > 1.0):
            taus[k] = 1.0
            if p.value is not None:
                ps[k] = p.value
        else:        
            taus[k] = opt
            ps[k] = p.value
    
    # Find minimizer
    min_ind = np.argmin(taus)
    tau_opt = taus[min_ind]
    p_opt = ps[min_ind]
    new_emp_cost =  (costs_precomputed @ p_opt).mean()
    
    return tau_opt, p_opt, taus, new_emp_cost

def optimize_quad_PAC_bound_bisection(costs_precomputed, p0, delta):
    '''Performs REP on the quadratic PAC-Bayes bound by sweeping on L_hat,
    bisectional search on lambda'''
    
    # Number of actions
    L = len(p0)
    
    # Number of environments
    m = np.shape(costs_precomputed)[0]
    
    C_bar = (1/m)*(np.ones((1,m)) @ costs_precomputed)
    min_cost = np.min(C_bar)
    max_cost = np.max(C_bar)
    
    L_hats = np.linspace(min_cost, max_cost, np.ceil((max_cost-min_cost)/0.001))
    R_p0 = np.log(2*np.sqrt(m)/delta)/(2*m)
    
    # Initialize vectors for storing optimal solutions
    tau_opt = ((C_bar@p0 + R_p0)**0.5 + R_p0**0.5)**2
    p_opt = p0
    
    for j in range(len(L_hats)):
        terminate = False
        L_hat = L_hats[j]
        min_lambda0 = (L_hat*R_p0 + R_p0**2)**0.5
        max_lambda0 = (tau_opt - L_hat)/2 - R_p0
        if min_lambda0 > max_lambda0:
            # If this happens then the prior gives a lower tau than any valid 
            # lambda choice
            terminate = True
        while not terminate:
            lambda0 = (min_lambda0 + max_lambda0)/2
            
            # Create cost function variable
            tau = cvx.Variable()
    
            # Create variable for probability vector
            p = cvx.Variable(L)
    
            cost_empirical = (1/m)*cvx.sum(costs_precomputed*p)
            R = (cvx.sum(cvx.kl_div(p, p0)) + np.log(2*np.sqrt(m)/delta))/(2*m)
            # R = 0.0
    
            # Constraints
            constraints = [tau >= L_hat + 2*R + 2*lambda0,
                            lambda0**2 >= L_hat*R + R**2, 
                            L_hat == cost_empirical, 
                            p >= 0, 
                            cvx.sum(p) == 1]

            prob = cvx.Problem(cvx.Minimize(tau), constraints)
    
            # Solve problem
            opt = prob.solve(verbose=False, solver=cvx.MOSEK) # , max_iters=3000)
            if np.isinf(opt) or (opt is None):
                min_lambda0 = lambda0
            else:
                max_lambda0 = lambda0
                
            if np.abs(lambda0 - (min_lambda0 + max_lambda0)/2) < 0.001:
                terminate = True
    
            # Store optimal value and optimizer
            if (opt < tau_opt):
                tau_opt = opt
                p_opt = p.value
    
    new_emp_cost = (costs_precomputed @ p_opt).mean()
    
    return tau_opt, p_opt, new_emp_cost

def kl_inverse(q, c):
    '''Compute kl inverse using Relative Entropy Programming'''    
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q,1-q])

    constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli,p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    prob.solve(verbose=False, solver=cvx.MOSEK) # solver=cvx.ECOS
    
    return p_bernoulli.value[0] 
    

if __name__ == "__main__":
    
    import argparse

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))
        return Collect_as

    parser = argparse.ArgumentParser(description='PAC-Bayes Optimization')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--num_envs', type=int, default=-1)
    parser.add_argument('--num_policies', type=int, default=-1)
    
    args = parser.parse_args()
    
    params = json.load(open(args.config_file))
    num_trials = args.num_envs
    num_actions = args.num_policies
    
    delta = params['delta']
    save_file_v = params['save_file_v']
    example = params['example']
    
    C = np.load("Weights/C_"+save_file_v+".npy")
    
    if num_trials == -1:
        num_trials = C.shape[0]
        num_actions = C.shape[1]
    
    C_emp = np.load("Weights/C_"+example+"_emp_test.npy")
    
    C = C[:num_trials,:num_actions]
    C_emp = C_emp[:,:num_actions]
    
    p0 = np.ones(num_actions)/num_actions

    print('========================')
    print('    Mc Allester Opt     ')
    print('========================')
    tau_opt, p_opt, taus, new_emp_cost = optimize_PAC_bound(C, p0, delta)
    print("McAllester PAC Bound:", tau_opt)
    print("New Emp Cost on Train Data:", new_emp_cost)
    r = (np.sum(scipy.special.kl_div(p_opt, p0)) + np.log(2*np.sqrt(num_trials)/delta))/(2*num_trials)
    print("R:",r)
    pac_bound = kl_inverse(new_emp_cost, 2*r)
    print("KL-inv PAC bound:", pac_bound)
    print("True Cost Estimate:", (C_emp @ p_opt).mean()) 
    
    print('========================')
    print('        Quad Opt        ')
    print('========================')
    tau_opt, p_opt, new_emp_cost = optimize_quad_PAC_bound_bisection(C, p0, delta)
    print("Quad PAC Bound:", tau_opt)
    print("New Emp Cost on Train Data:", new_emp_cost)
    r = (np.sum(scipy.special.kl_div(p_opt, p0)) + np.log(2*np.sqrt(num_trials)/delta))/(2*num_trials)
    print("R:",r)
    quad_pac_bound = kl_inverse(new_emp_cost, 2*r)
    print("KL-inv PAC bound:", quad_pac_bound)
    print("True Cost Estimate:", (C_emp @ p_opt).mean()) 

    if quad_pac_bound < pac_bound:
        np.save("Weights/p_"+save_file_v+".npy", p_opt)