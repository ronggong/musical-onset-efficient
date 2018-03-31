import numpy as np
from scipy.stats import norm
from src.parameters_jingju import hopsize_t
cimport cython
# from numpy.math cimport INFINITY

value_eps = np.finfo(float).eps

def FdurationProba2( syllable_duration, param_s ):

    M1      = syllable_duration/hopsize_t

    # % delta
    if param_s['delta_mode'] == 'constant':
        delta   = param_s['delta']
    elif param_s['delta_mode'] == 'proportion':
        delta   = syllable_duration * param_s['delta']
    else:
        raise('Error: delta_default should be either constant or proportion.')
    S1          = delta/hopsize_t

    # % duration max is three times of standard deviation
    duration_max = syllable_duration + 3.0*delta

    tmin = 0

    tmax = int(duration_max/hopsize_t)

    # Ps = pdf('Normal',(tmin : tmax), M1, S1)
    x = range(tmin, tmax)
    Ps = norm.pdf(x, M1, S1)
    return Ps, tmin, tmax

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def viterbiSegmental2(P, sd, param_s):
    """
    :param P: NxT emission probability state sequence (P(j,t) = emission probability of symbol j at time t)
    :param sd: 1xT score duration array
    :param param_s:
    :return:
    """

    # preventsingularities
    P[P == 0]     = value_eps

    i_bound = np.where(P > value_eps)[0]
    N = len(i_bound)
    T = len(sd)

    # log - likelihood
    delta   = np.zeros((N, T), dtype=np.double)
    psi     = np.zeros((N, T), dtype=np.double)
    logP    = np.log(P, dtype=np.double)

    # duration probability
    Ps, _, _  = FdurationProba2(sd[0], param_s)
    Ps[Ps == 0]     = value_eps
    C               = len(Ps)
    logPs           = np.log(Ps, dtype=np.double)

    cdef double [:, ::1] cdelta = delta
    cdef double [:, ::1] cpsi   = psi
    cdef double [::1] clogP     = logP
    cdef double [::1] clogPs    = logPs
    cdef int [::1] ci_bound     = np.array(i_bound, dtype=np.intc)
    # % % % % % % % % % % % % % % % % % %
    # % Initialisation %
    # % % % % % % % % % % % % % % % % % %

    # % not a possible transition from > 0 time to 1
    cdelta[0, 0]     = -np.inf
    cpsi[:,0]        = 0
    for jj in range(1,N):
        d = ci_bound[jj] - ci_bound[0]
        # print(jj, i_bound[jj], d, C)
        if d >= C:
            cdelta[jj, 0] = -np.inf
        else:
            cdelta[jj, 0] = clogPs[d] + clogP[i_bound[jj]]

    clogPs = None

    # % % % % % % % % % % % % % % % % % %
    # % Recursion %
    # % % % % % % % % % % % % % % % % % %
    delta_current = np.zeros((N,), dtype=np.double)
    cdef double [::1] cdelta_current = delta_current

    for t in range(1,T - 1):
        print(t)
        # % duration probability
        Ps, _, _  = FdurationProba2(sd[t], param_s)
        Ps[Ps == 0]     = value_eps
        C               = len(Ps)
        logPs           = np.log(Ps, dtype=np.double)

        for jj in range(N):
            for ii in range(N):
                # print(i_bound, jj, ii)
                d = ci_bound[jj] - ci_bound[ii]
                # print(d, C)
                if d >= C or d <= 0:
                    cdelta_current[ii] = -np.inf
                else:
                    cdelta_current[ii] = cdelta[ii, t - 1] + logPs[d]

            I_delta             = np.argmax(delta_current)
            M_delta             = delta_current[I_delta]
            cdelta[jj, t]        = M_delta + clogP[i_bound[jj]]
            cpsi[jj, t]          = I_delta

    # % duration probability
    Ps, tmin, tmax  = FdurationProba2(sd[T-1], param_s)
    Ps[Ps == 0]     = value_eps
    C               = len(Ps)
    logPs           = np.log(Ps, dtype=np.double)
    clogPs          = logPs
    # delta_current   = np.zeros((N,))

    for ii in range(N):
        d = ci_bound[N-1] - ci_bound[ii]
        if d >= C or d <= 0:
            cdelta_current[ii] = -np.inf
        else:
            cdelta_current[ii] = cdelta[ii, T-2] + clogPs[d]
    I_delta             = np.argmax(delta_current)
    M_delta             = delta_current[I_delta]
    cdelta[N-1, T-1]        = M_delta + clogP[i_bound[N-1]]
    cpsi[N-1, T-1]          = I_delta

    # % % % % % % % % % % % % % % % % % %
    # % Backtrack %
    # % % % % % % % % % % % % % % % % % %
    i_best_sequence = np.zeros((T+1,),dtype=int)
    # print(i_best_sequence)
    i_best_sequence[T] = N-1
    for t in range(T)[::-1]:
        # print(t+1, i_best_sequence[t+1])
        i_best_sequence[t] = int(cpsi[int(i_best_sequence[t + 1]), t])
    # print(i_best_sequence)
    i_boundary = [i_bound[ii] for ii in i_best_sequence]

    cdelta          = None
    cdelta_current  = None
    clogPs          = None
    clogP           = None
    cpsi            = None
    ci_bound        = None

    return i_boundary