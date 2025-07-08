import cma

class CMAESSolverParams():
    def __init__(self, maxiter=10, sigma=1.0, popsize=10, seed=0, num_processes=1):
        self.maxiter = maxiter
        self.popsize = popsize
        self.seed = seed
        self.sigma = sigma
        self.num_processes = num_processes
class CMAESSolver():
    
    def __init__(self, objective_func,  cmaes_params):
        self.cmaes_params = cmaes_params
        self.objective_func = objective_func
        
    
    def solve(self, x, return_result=False, return_history=False):# cannot be an instance method
        # linear transformation of the parameters to lie between -1, when p=p_min and 1 when p=p_max. It's equal to 0 if p=p0
    
        es = cma.CMAEvolutionStrategy( x, self.cmaes_params.sigma, 
                                      {'maxiter':self.cmaes_params.maxiter, 
                                       'popsize':self.cmaes_params.popsize, 
                                       'seed':self.cmaes_params.seed})
        
        if return_history:
            running_history = []
        
        if self.cmaes_params.num_processes > 1:
            from cma.optimization_tools import EvalParallel2
            # use with `with` statement (context manager)
            # cma.CMAOptions
            with EvalParallel2(self.objective_func, number_of_processes= self.cmaes_params.num_processes) as eval_all:
                while not es.stop():
                    X = es.ask()
                    es.tell(X, eval_all(X))
                    es.disp()
                    
                    # save result to running history every iteration
                    
                    if return_history:
                        running_history.append(es.result._asdict())   
                    
        else:
            while not es.stop():
                X = es.ask()
                objs = [0]*len(X)
                for i in range(len(X)):
                    objs[i] = self.objective_func(X[i])
                
                es.tell(X,objs)
                es.disp()
                running_history.append(es.result._asdict())
                
        output = es.result.xbest
        if return_result:
            output = (output, es.result._asdict())

        if return_history:
            # if output is a tuple, concatenate the running history to the output
            if isinstance(output, tuple):
                output = output + (running_history,)
            else:
                output = (output, running_history)
        return output



