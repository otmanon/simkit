from .gradient_cfd import gradient_cfd


def hessian_cfd(phi, y, h):
    """
    Computes the finite difference gradient of a function f, with respect to each of the parameters y
    if phi spits out a tensor of shape (n1, n2, ..., nd), then gradient_cfd will spit out a tensor of shape (dim_1, dim_2, ..., dim_n, dim(y))
    """

    def grad_func(p):
        return gradient_cfd(phi, p, h)

    phi_hess_fd = gradient_cfd(grad_func, y, h)
    return phi_hess_fd