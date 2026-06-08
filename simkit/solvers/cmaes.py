try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False


def cmaes(x, objective_func, maxiter=10, sigma=1.0, popsize=10, seed=0,
                 num_processes=1, return_result=False, return_history=False):
    """
    CMA-ES solver for derivative-free minimization.

    Parameters
    ----------
    x : array-like
        Initial guess for the parameters.
    objective_func : callable
        Maps a parameter vector to a scalar objective value.
    maxiter : int
        Maximum number of generations.
    sigma : float
        Initial standard deviation of the search distribution.
    popsize : int
        Population size per generation.
    seed : int
        Random seed.
    num_processes : int
        Number of processes used to evaluate the objective in parallel.
    return_result : bool
        If True, also return the full CMA-ES result dict.
    return_history : bool
        If True, also return the per-generation result history.

    Returns
    -------
    output : np.ndarray or tuple
        Best parameters found, optionally bundled with the result dict and/or
        history depending on ``return_result`` / ``return_history``.
    """
    if not CMA_AVAILABLE:
        raise ImportError(
            "cmaes_solver requires the 'cma' package. "
            "Install it with: pip install -e .[cmaes] or pip install cma"
        )

    es = cma.CMAEvolutionStrategy(x, sigma,
                                  {'maxiter': maxiter,
                                   'popsize': popsize,
                                   'seed': seed})

    if return_history:
        running_history = []

    if num_processes > 1:
        from cma.optimization_tools import EvalParallel2
        # use with `with` statement (context manager)
        with EvalParallel2(objective_func, number_of_processes=num_processes) as eval_all:
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
            objs = [0] * len(X)
            for i in range(len(X)):
                objs[i] = objective_func(X[i])

            es.tell(X, objs)
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
