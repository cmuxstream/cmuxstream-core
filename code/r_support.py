import numpy as np
import pickle
import logging
import random
import os
import os.path
import errno
from pandas import DataFrame
import sys
from timeit import default_timer as timer
from datetime import timedelta
import traceback

from statsmodels.distributions.empirical_distribution import ECDF
import ranking
from ranking import Ranking

from scipy.sparse import csr_matrix
import scipy.stats as stats
import scipy.optimize as opt

from sklearn.linear_model import LogisticRegression as LR

import cvxopt


"""
Some R-like functions to help port R code to python
"""


logger = logging.getLogger(__name__)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed + 32767)


def matrix(d, nrow=None, ncol=None, byrow=False):
    """Returns the data as a 2-D matrix

    A copy of the same matrix will be returned if input data dimensions are
    same as output data dimensions. Else, a new matrix will be created
    and returned.

    Example:
        d = np.reshape(range(12), (6, 2))
        matrix(d[0:2, :], nrow=2, byrow=True)

    Args:
        d:
        nrow:
        ncol:
        byrow:

    Returns: np.ndarray
    """
    if byrow:
        # fill by row...in python 'C' fills by the last axis
        # therefore, data gets populated one-row at a time
        order = 'C'
    else:
        # fill by column...in python 'F' fills by the first axis
        # therefore, data gets populated one-column at a time
        order = 'F'
    if len(d.shape) == 2:
        d_rows, d_cols = d.shape
    elif len(d.shape) == 1:
        d_rows, d_cols = (1, d.shape[0])
    else:
        raise ValueError("Dimensions more than 2 are not supported")
    if nrow is not None and ncol is None:
        ncol = int(d_rows * d_cols / float(nrow))
    elif ncol is not None and nrow is None:
        nrow = int(d_rows * d_cols / float(ncol))
    if len(d.shape) == 2 and d_rows == nrow and d_cols == ncol:
        return d.copy()
    if not d_rows * d_cols == nrow * ncol:
        raise ValueError("input dimensions (%d, %d) not compatible with output dimensions (%d, %d)" %
                         (d_rows, d_cols, nrow, ncol))
    if isinstance(d, csr_matrix):
        return d.reshape((nrow, ncol), order=order)
    else:
        return np.reshape(d, (nrow, ncol), order=order)


# Ranks in decreasing order
def rank(x, ties_method="average"):
    ox = np.argsort(-x)
    sx = np.argsort(ox)
    if ties_method == "average":
        strategy = ranking.FRACTIONAL
    else:
        strategy = ranking.COMPETITION
    r = Ranking(x[ox], strategy=strategy, start=1)
    rnks = list(r.ranks())
    return np.array(rnks)[sx]


def nrow(x):
    if len(x.shape) == 2:
        return x.shape[0]
    return None


def ncol(x):
    if len(x.shape) == 2:
        return x.shape[1]
    return None


def rbind(m1, m2):
    if m1 is None:
        return np.copy(m2)
    return np.append(m1, m2, axis=0)


def cbind(m1, m2):
    if len(m1.shape) == 1 and len(m2.shape) == 1:
        if len(m1) == len(m2):
            mat = np.empty(shape=(len(m1), 2))
            mat[:, 0] = m1
            mat[:, 1] = m2
            return mat
        else:
            raise ValueError("length of arrays differ: (%d, %d)" % (len(m1), len(m2)))
    return np.append(m1, m2, axis=1)


def sample(x, n):
    shuffle = np.array(x)
    np.random.shuffle(shuffle)
    return shuffle[0:n]


def append(a1, a2):
    if isinstance(a1, np.ndarray) and len(a1.shape) == 1:
        return np.append(a1, a2)
    a = a1[:]
    if isinstance(a2, list):
        a.extend(a2)
    else:
        a.append(a2)
    return a


def rep(val, n, dtype=float):
    return np.ones(n, dtype=dtype) * val


def quantile(x, q):
    return np.percentile(x, q)


def difftime(endtime, starttime, units="secs"):
    if units == "secs":
        t = timedelta(seconds=endtime-starttime)
    else:
        raise ValueError("units '%s' not supported!" % (units,))
    return t.seconds


def order(x, decreasing=False):
    if decreasing:
        return np.argsort(-x)
    else:
        return np.argsort(x)


def runif(n, min=0.0, max=1.0):
    return stats.uniform.rvs(loc=min, scale=min+max, size=n)


def rnorm(n, mean=0.0, sd=1.0):
    return stats.norm.rvs(loc=mean, scale=sd, size=n)


def pnorm(x, mean=0.0, sd=1.0):
    return stats.norm.cdf(x, loc=mean, scale=sd)


def ecdf(x):
    return ECDF(x)


def matrix_rank(x):
    return np.linalg.matrix_rank(x)


class LogisticRegressionClassifier(object):
    """
    see:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    def __init__(self):
        self.lr = None

    @staticmethod
    def fit(x, y):
        classifier = LogisticRegressionClassifier()
        classifier.lr = LR(penalty='l2', dual=False, tol=0.0001, C=1,
                           fit_intercept=True, intercept_scaling=1,
                           class_weight=None, random_state=None, solver='liblinear',
                           max_iter=100, multi_class='ovr', verbose=0)
        classifier.lr.fit(x, y)
        return classifier

    def predict(self, x, type="response"):
        if self.lr is None:
            raise ValueError("classifier not initialized/trained...")
        if type == "response":
            y = self.lr.predict_proba(x)
        else:
            y = self.lr.predict(x)
        return y

    def predict_prob_for_class(self, x, cls):
        if self.lr is None:
            raise ValueError("classifier not initialized/trained...")
        clsindex = np.where(self.lr.classes_ == cls)[0][0]
        # logger.debug("class index: %d" % (clsindex,))
        y = self.lr.predict_proba(x)[:, clsindex]
        return y


def read_csv(file, header=None, sep=','):
    """Loads data from a CSV

    Returns:
        DataFrame
    """

    if header is not None and header:
        header = 0 # first row is header

    data_df = DataFrame.from_csv(file, header=header, sep=sep, index_col=None)

    #datamat = np.ndarray(shape=data_df.shape, dtype=float)
    #datamat[:, :] = data_df.iloc[:, 0:data_df.shape[1]]

    return data_df


def save(obj, filepath):
    filehandler = open(filepath, 'w')
    pickle.dump(obj, filehandler)
    return obj


def load(filepath):
    filehandler = open(filepath, 'r')
    obj = pickle.load(filehandler)
    return obj


def dir_create(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def exception_to_string(exc):
    exc_type, exc_value, exc_traceback = exc
    return (str(exc_type) + os.linesep + str(exc_value)
            + os.linesep + str(traceback.extract_tb(exc_traceback)))


def configure_logger(args):
    global logger
    logger_format = "%(levelname)s [%(asctime)s]: %(message)s"
    logger_level = logging.DEBUG if args.debug else logging.ERROR
    if args.log_file is not None and args.log_file != "":
        # print "configuring logger to file %s" % (args.log_file,)
        logging.basicConfig(filename=args.log_file,
                            level=logger_level, format=logger_format,
                            filemode='w') # use filemode='a' for APPEND
    else:
        logging.basicConfig(level=logger_level, format=logger_format)
    logger = logging.getLogger("default")


class Timer(object):
    def __init__(self):
        self.start_time = timer()
        self.end_time = None

    def start(self):
        self.start_time = timer()
        self.end_time = None

    def end(self):
        self.end_time = timer()

    def elapsed(self):
        etime = self.end_time
        if etime is None:
            etime = timer()
        return difftime(etime, self.start_time, units="secs")

    def message(self, msg):
        if self.end_time is None:
            self.end_time = timer()
        tdiff = self.elapsed()
        return "%s %f sec(s)" % (msg, tdiff)


def constr_optim(theta, f, grad=None, ui=None, ci=None, a=None, b=None,
                 hessian=None, bounds=None, method="BFGS",
                 outer_iterations=500, debug=False, args=None):
    """solve non-linear constraint optimization with scipy.optimize

    problems have the form:
        minimize f_0(x)
        s.t.
            ui * x >= ci             --> Note: this is opposite of cvxopt
            a * x = b                --> Supported
            #f_k(x) <= 0, k=1..m     --> Not supported

    :param theta: np.array
            initial values. Must be in the domain of f()
    :param f: function that is being minimized
            returns the function evaluation
    :param grad: function
            returns the first derivative
    :param ui: np.ndarray
    :param ci: np.array
    :param a: np.ndarray
    :param b: np.array
    :param mu:
    :param control:
    :param method:
    :param hessian:
    :param outer_iterations:
    :param outer_eps:
    :param debug:
    :param bounds:
    :param args:
    :return:
    """
    x0 = np.array(theta)
    # build the constraint set
    cons = ()
    if ui is not None:
        for i in range(nrow(ui)):
            # cons += ({'type': 'ineq', 'fun': lambda x: x.dot(u_) - c_},)
            def fcons_ineq(x, i=i):
                return x.dot(ui[i, :]) - ci[i]
            cons += ({'type': 'ineq', 'fun': fcons_ineq},)
    if a is not None:
        for i in range(nrow(a)):
            def fcons_eq(x, i=i):
                return x.dot(a[i, :]) - b[i]
            cons += ({'type': 'eq', 'fun': fcons_eq},)
    res = opt.minimize(f, x0,
                       args=() if args is None else args,
                       method=method, jac=grad,
                       hess=hessian, hessp=None, bounds=bounds,
                       constraints=cons, tol=1e-6, callback=None,
                       #options={'gtol': 1e-6, 'maxiter': outer_iterations, 'disp': True}
                       options={'maxiter': outer_iterations, 'disp': debug}
                       )
    if not res.success:
        logger.debug("Optimization Failure:\nStatus: %d; Msg: %s" % (res.status, res.message))
    return res.x, res.success


def get_box_constraints(n, bounds=None):
    box_lims = np.empty(shape=(n, 2))
    box_lims[:, 0] = -np.Inf
    box_lims[:, 1] = np.Inf
    if bounds is not None:
        for i in range(n):
            mn, mx = bounds[i]
            mn = -np.Inf if mn is None else mn
            mx = np.Inf if mx is None else mx
            box_lims[i, 0] = mn
            box_lims[i, 1] = mx
    return box_lims


def get_kktsolver_no_equality_constraints(ui=None, fn=None, grad=None, hessian=None, debug=False):
    """ Returns the kktsolver

    :param ui: np.ndarray
        ui = -G for CVXOPT
    :param fn:
    :param grad:
    :param hessian:
    :param debug:
    :return:
    """

    # Note that we negate ui because in other optimization
    # APIs we follow the convention that G.x >= h whereas CVXOPT uses G.x <= h
    G = cvxopt.matrix(-ui, ui.shape) if ui is not None else None

    def kktsolver(x, z, W):
        """KKT solver for the specific case when there are no equality constraints

        problem is:
            minimize f(x)
            s.t.
                G.x <= h
            where G = -ui

            The KKT equations are solutions of:

            [  H       G_tilde'  ] [ux]   [bx]
            [                    ] [  ] = [  ]
            [  G_tilde  -W'W     ] [uz]   [bz]

            G_tilde = [G']' = G (in case there are no non-linear constraints, like in AAD)

            Simplifying:
            [  H    G'  ] [ux]   [bx]
            [           ] [  ] = [  ]
            [  G  -W'W  ] [uz]   [bz]

            Upon solution, the last component bz must be scaled, i.e.: bz := W.uz

            To solve:
                Let:
                    P = G'(W'W)^(-1)G
                    S = G'(W'W)^(-1)
                    Q = H + P
                v = bx + S.bz
                Q.ux = v
                W.uz = (W')^(-1)(G.ux - bz)
        """

        if debug:
            logger.debug("Setup kkt solver")
            logger.debug("W")
            for key in W.keys():
                logger.debug("key: %s" % (key,))
                logger.debug(W[key])

        H = hessian(x)
        if debug:
            logger.debug("diag H")
            logger.debug(np.diag(H))
        _H = cvxopt.spdiag(list(np.diag(H))) if H is not None else None

        wdi = W["di"]
        Wdi2 = cvxopt.spdiag(cvxopt.mul(wdi, wdi))

        S = G.T * Wdi2
        P = S * G

        Q = _H + P
        # now, do the cholesky decomposition of Q
        cvxopt.lapack.potrf(Q)

        if False and fn is not None:
            logger.debug("At setup f(x) = %d" % (fn(np.array(list(x))),))

        def f(x, y, z):
            if False and fn is not None:
                logger.debug("f(x) = %d" % (fn(np.array(list(x))),))
            try:
                # logger.debug("Compute x := S * z + x...")
                cvxopt.blas.gemv(S, z, x, alpha=1.0, beta=1.0)  # x = S * z + x
                cvxopt.lapack.potrs(Q, x)
            except BaseException, e:
                logger.debug(exception_to_string(sys.exc_info()))
                raise e
            cvxopt.blas.gemv(G, x, z, alpha=1.0, beta=-1.0)  # z = _G * x - z
            z[:] = cvxopt.mul(wdi, z)  # scaled z
            # raise NotImplementedError("Method Not implemented yet")
        return f

    return kktsolver


def cvx_optim(theta, f, grad=None, ui=None, ci=None, a=None, b=None,
              hessian=None, bounds=None, method="BFGS", kktsolver=None,
              outer_iterations=100, debug=False, args=None):
    """Uses CVXOPT library for optimization

    The general form of the optimization problem is:

    minimize f(x)
    s.t
        ui * x >= ci  <- This is different from the CXOPT API. We will be switching the sign before calling CVXOPT
        a * x  = b
        # f_k(x) <= 0, k=1,...,m  <-- not supported

    :param theta: numpy.array
        initial values
    :param f: function
    :param grad: function
    :param ui: numpy.ndarray
    :param ci: numpy.array
    :param a: numpy.ndarray
    :param b: numpy.array
    :param method: string
    :param kktsolver: string
    :param hessian: function
    :param outer_iterations:
    :param debug: boolean
    :param bounds: not supported
    :param args:
    :return:
    """
    n = len(theta)

    x0 = cvxopt.matrix(theta, (n, 1))

    box_lims = get_box_constraints(n, bounds)
    # logger.debug(box_lims)

    def F(x=None, z=None):
        if x is None:
            return 0, x0
        if bounds is not None:
            for j in range(n):
                v = x[j, 0]
                if v < box_lims[j, 0] or v > box_lims[j, 1]:
                    return None

        # convert the variable to numpy that will be understood by the caller.
        m = x.size[0]
        xx = np.array([x[j, 0] for j in range(m)])

        Df = cvxopt.matrix(grad(xx), (1, n))
        if z is None:
            return f(xx), Df
        if True:
            H = z[0] * cvxopt.matrix(hessian(xx))
        else:
            # *ONLY* in case we are *sure* that the hessian is diagonal
            _H = np.diag(hessian(xx))
            H = z[0] * cvxopt.spdiag(list(_H))
        return f(xx), Df, H

    A = None
    bx = None
    G = None
    h = None
    if a is not None:
        A = cvxopt.matrix(a, a.shape)
        bx = cvxopt.matrix(b, (len(b), 1))
    if ui is not None:
        G = -cvxopt.matrix(ui, ui.shape)
        h = -cvxopt.matrix(ci, (len(ci), 1))

    if False:
        logger.debug("A: %s" % ("" if b is None else str(A.size),))
        # logger.debug(A)
        logger.debug("b: %s" % ("" if b is None else str(b.size),))
        # logger.debug(bx)
        logger.debug("G: %s" % str(G.size))
        logger.debug(G)
        logger.debug("h: %s" % str(h.size))
        logger.debug(h)

    # cvxopt.solvers.options['show_progress'] = False
    options = {'show_progress': False, 'maxiters': outer_iterations}
    soln = cvxopt.solvers.cp(F, G=G, h=h, A=A, b=bx, options=options, kktsolver=kktsolver)
    # logger.debug(soln.keys())
    # logger.debug(soln['status'])
    sx = soln['x']
    x = np.array([sx[i, 0] for i in range(n)])
    success = soln['status'] == 'optimal'

    # if False and not success:
    if False:
        logger.debug("A:")
        logger.debug(A)
        logger.debug("b:")
        logger.debug(bx)
        logger.debug("G:")
        logger.debug(G)
        logger.debug("h:")
        logger.debug(h)
        fx, Dfx, hessx = F(sx, [1])
        logger.debug("f(x)")
        logger.debug(fx)
        logger.debug("g(x)")
        logger.debug(Dfx)
        logger.debug("hessian(x)")
        logger.debug(np.array(hessx))
        logger.debug(soln.keys())
        for key in soln.keys():
            logger.debug("key: %s" % (key,))
            logger.debug(soln[key])

    return x, success

