def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-6, norm=Inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False, error_list=False,
                   **unknown_options):

    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    func_calls, f = wrap_function(f, args)

    old_fval = f(x0)

    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = numpy.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1

    #old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k = armijo_search(f, xk, pk, gfk)
            if alpha_k == None:
                warnflag = 2
                break
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        #if gfkp1 is None:
        #    gfkp1 = myfprime(xkp1)

        gfkp1 = myfprime(xkp1)
        yk = gfkp1 - gfk

        sTy = np.dot(sk.T, yk)
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm) > 1e-2:
            w = 2
        else:
            w = 100

        if sTy < 0:
            sTs = np.dot(sk.T, sk)
            xhi = w - (sTy / (sTs * gnorm))
        else:
            xhi = w
        #snorm = vecnorm(sk, ord=norm)

        #xhi = w*gnorm + numpy.maximum([-numpy.dot(yk, sk)/ numpy.square(snorm)],0)

        yk1 = yk + xhi*gnorm*sk
        gfk = gfkp1


        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not numpy.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk1, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok):  # this is patch for numpy
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, numpy.newaxis] * yk1[numpy.newaxis, :] * rhok
        A2 = I - yk1[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
        Hk = numpy.dot(A1, numpy.dot(Hk, A2)) + (rhok * sk[:, numpy.newaxis] *
                                                 sk[numpy.newaxis, :])
        old_fval = f(xk)
        error_list.append(old_fval)

    fval = old_fval
    if np.isnan(fval):
        # This can happen if the first call to f returned NaN;
        # the loop is then never entered.
        warnflag = 2

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % func_calls[0])
        print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result

def armijo_search(f, xk, pk, gfk):

    alpha = 0.9
    func1 = f(xk)
    func2 = numpy.dot(gfk.T, pk)
    for i in range(10):
        if f(xk + alpha*pk) > func1 + 0.01 * alpha * func2:
            alpha = alpha/2
        else:
            return alpha
    return None
