"""
-- APPENDIX A --
Implemented methods for AC-FE and NLA, reused in various questions
@author: Rong Feng <rmfeng@gmail.com>
"""


import numpy as np
from scipy.linalg import solve
from scipy.linalg import cho_solve

DEFAULT_THRESH = 10 ** -12

"""
-- Section 1 -- Black Scholes
"""


def f_1(x):
    return np.exp(-(x ** 2) / 2)


def norm_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / 2)


def i_simpson(a, b, n, fofx):
    """ numeric implementation of the simpson's rule """
    h = (b - a) / n
    i_simp = fofx(a) / 6 + fofx(b) / 6
    for i in range(1, n):
        i_simp += fofx(a + i * h) / 3
    for i in range(1, n + 1):
        i_simp += 2 * fofx(a + (i - 0.5) * h) / 3
    return h * i_simp


def norm_cdf(x, n):
    """ using the simpson's rull with n intervals, estimate the normal cdf """
    if x > 0:
        return 0.5 + (1 / np.sqrt(2 * np.pi)) * i_simpson(0, x, n, f_1)
    elif x < 0:
        return 0.5 - (1 / np.sqrt(2 * np.pi)) * i_simpson(x, 0, n, f_1)
    else:
        return 0.5


def norm_cdf_thresh(x, thresh):
    """ given a threshold, will return the normal cdf using simpson's rule to an
    accuracy close of atleast that threshold
    """
    n_0, n = 4, 8
    i_old, i_new = norm_cdf(x, n_0), norm_cdf(x, n)
    while (np.abs(i_new - i_old) > thresh):
        i_old = i_new
        n = 2 * n
        i_new = norm_cdf(x, n)
    return i_new


def norm_cdf_def_thresh(x):
    """ returns the normal cdf using the default threshold specificed in the sheet """
    return norm_cdf_thresh(x, DEFAULT_THRESH)


def bs_price(T, isCall, S, K, vol, r, q, n_cdf):
    """ black scholes price """
    d1 = (np.log(S / K) + (r - q + (vol ** 2) / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - (vol * np.sqrt(T))

    if isCall:
        # call
        return S * np.exp(-q * T) * n_cdf(d1) - K * np.exp(-r * T) * n_cdf(d2)
    else:
        # put
        return K * np.exp(-r * T) * n_cdf(-d2) - S * np.exp(-q * T) * n_cdf(-d1)


def bs_delta(T, isCall, S, K, vol, r, q, n_cdf):
    d1 = (np.log(S / K) + (r - q + (vol ** 2) / 2) * T) / (vol * np.sqrt(T))
    if isCall:
        return n_cdf(d1)
    else:
        return -n_cdf(-d1)


def bs_gamma(T, S, K, vol, r, q):
    d1 = (np.log(S / K) + (r - q + (vol ** 2) / 2) * T) / (vol * np.sqrt(T))
    return norm_pdf(d1) / (S * vol * np.sqrt(T))


def bs_theta(T, isCall, S, K, vol, r, q, n_cdf):
    d1 = (np.log(S / K) + (r - q + (vol ** 2) / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - (vol * np.sqrt(T))
    numerator = -S * norm_pdf(d1) * vol
    denominator = 2 * np.sqrt(T)

    if isCall:
        term_2 = -r * K * np.exp(-r*T) * n_cdf(d2)
    else:
        term_2 = r * K * np.exp(-r*T) * n_cdf(-d2)

    return numerator / denominator + term_2


def bs_vega(T, S, K, vol, r, q):
    """ black scholes vega """
    d1 = (np.log(S / K) + (r - q + (vol ** 2) / 2) * T) / (vol * np.sqrt(T))
    return S * np.exp(-q*T) * norm_pdf(d1) * np.sqrt(T)


def imp_vol_newton(px, v_guess, T, isCall, S, K, r, q, n_cdf, tol=10**-6, max_iter=100, is_verbose=False):
    """ uses newton's method to solve for implied vol """
    cur_iter = 0
    cur_vol = v_guess
    cur_f_val = bs_price(T, isCall, S, K, cur_vol, r, q, n_cdf) - px
    if is_verbose:
        print("initial guessed vol price =", cur_f_val + px)

    while cur_iter < max_iter and np.abs(cur_f_val) > tol:
        if is_verbose:
            print("Not close enough, doing next iteration ...")
        cur_vega = bs_vega(T, S, K, cur_vol, r, q)
        if is_verbose:
            print("current vega =", cur_vega)
        cur_vol = cur_vol - (cur_f_val / cur_vega)
        if is_verbose:
            print("new vol =", cur_vol)
        cur_f_val = bs_price(T, isCall, S, K, cur_vol, r, q, n_cdf) - px
        if is_verbose:
            print("new price =", cur_f_val + px, "\n")
        cur_iter += 1

    return cur_vol

"""
-- Section 2 -- Bond Math
"""


def df_fn(t, r_of_t):
    return np.exp(-r_of_t(t) * t)


def df_yld(t, yld):
    return np.exp(-yld * t)


def gen_cf(t_list, cpn_rate, cpn_per_y, par=100):
    cf = cpn_rate * par / cpn_per_y
    total_len = len(t_list)
    cf_list = [0] + [cf] * (total_len - 1)
    cf_list[-1] += par
    return cf_list


def gen_t_list(mat_in_m, freq):
    interval_in_m = 12 / freq
    return np.arange((mat_in_m % interval_in_m)/12, (mat_in_m + 1)/12, interval_in_m / 12)[0:]


def price_bond(yld, mat_in_m, cpn_rate, freq):
    """ prices a bullet bond given yield"""
    t_list = gen_t_list(mat_in_m, freq)
    cf_list = gen_cf(t_list, cpn_rate, freq)
    return price_bond_w_lists(yld, t_list, cf_list)


def price_bond_w_lists(yld, t_list, cf_list):
    """ prices a bond given list of times and list of cfs """
    assert len(t_list) == len(cf_list)
    df_list = df_yld(t_list, yld)
    return np.sum(cf_list * df_list)


def bond_yield_deriv(yld, mat_in_m, cpn_rate, freq):
    t_list = gen_t_list(mat_in_m, freq)
    cf_list = gen_cf(t_list, cpn_rate, freq)
    return bond_yield_deriv_w_lists(yld, t_list, cf_list)


def bond_yield_deriv_w_lists(yld, t_list, cf_list):
    assert len(t_list) == len(cf_list)
    df_list = df_yld(t_list, yld)
    return np.sum(cf_list * df_list * t_list)


def price_bond_w_dur_cvx(t_list, cf_list, y):
    """
    T = time to maturity in years
    n = number of cfs
    t_list = vector of cf dates in years
    cf_list = vector of cf amounts in years
    y = yield of the bond
    """
    assert (len(t_list) == len(cf_list))

    price, duration, convexity = 0, 0, 0
    for i in range(0, len(t_list)):
        cur_df = df_yld(y, t_list[i])
        price += cf_list[i] * cur_df
        duration += t_list[i] * cf_list[i] * cur_df
        convexity += (t_list[i] ** 2) * cf_list[i] * cur_df

    return price, duration / price, convexity / price


def ytm_newton(yld_guess, px, mat_in_m, cpn_rate, freq, tol_consec=10**-6, max_iter=100):
    """ Uses Newton's method to compute the yield of a bond """
    cur_iter = 0
    cur_yld = yld_guess

    # these lists remain the same at each iteration
    t_list = gen_t_list(mat_in_m, freq)
    cf_list = gen_cf(t_list, cpn_rate, freq)

    cur_f_val = price_bond_w_lists(cur_yld, t_list, cf_list) - px
    cur_chg = cur_f_val
    print("initial guessed yield implied price =", cur_f_val + px)

    while cur_iter < max_iter and np.abs(cur_chg) > tol_consec:
        print("Not close enough, doing next iteration ...")
        cur_deriv = bond_yield_deriv_w_lists(cur_yld, t_list, cf_list)
        print("current yield =", cur_yld)
        cur_yld = cur_yld + (cur_f_val / cur_deriv)
        print("new yield =", cur_yld)
        prev_f_val = cur_f_val
        cur_f_val = price_bond_w_lists(cur_yld, t_list, cf_list) - px
        cur_chg = (cur_f_val - prev_f_val)
        print("new price =", cur_f_val + px, "\n")
        cur_iter += 1

    return cur_yld


def generic_newton(x_guess, f_of_x, fprime_of_x, tol_consec=10**-6, max_iter=100, is_verbose=True):
    """ Uses newton's method to find the 0, provide a function and its derivative """
    cur_iter = 0
    cur_x = x_guess

    cur_f_val = f_of_x(cur_x)
    cur_chg = cur_f_val
    if is_verbose: print("f(initial guess) =", cur_f_val)

    while cur_iter < max_iter and np.abs(cur_chg) > tol_consec:
        if is_verbose: print("Not close enough, doing next iteration: %s" % str(cur_iter + 1))
        cur_deriv = fprime_of_x(cur_x)
        cur_x = cur_x - (cur_f_val / cur_deriv)
        if is_verbose: print("new x =", cur_x)
        prev_f_val = cur_f_val
        cur_f_val = f_of_x(cur_x)
        if is_verbose: print("new f(x) =", cur_f_val)
        cur_chg = (cur_f_val - prev_f_val)
        if is_verbose: print("f(x) change this iteration =", cur_chg, "\n")
        cur_iter += 1

        if is_verbose: print("zero was found after %s iterations ... " % cur_iter)
    return cur_x


""" --- CUBIC SPLINE --- """


def efficient_cub_spline(x, v):
    n = len(x) - 1
    z = np.array([0.0] * (n - 1))

    for i in range(1, n):
        z[i - 1] = 6 * ((v[i + 1] - v[i]) / (x[i + 1] - x[i]) - (v[i] - v[i - 1]) / (x[i] - x[i - 1]))

    # making blank M array
    M = np.array([[0.0] * (n - 1)])
    for i in range(1, n - 1):
        M = np.append(M, np.array([[0.0] * (n - 1)]), axis=0)

    # updating M
    for i in range(1, n):
        M[i - 1, i - 1] = 2 * (x[i + 1] - x[i - 1])

    for i in range(1, n - 1):
        M[i - 1, i] = x[i + 1] - x[i]

    for i in range(2, n):
        M[i - 1, i - 2] = x[i] - x[i - 1]

    # solving
    s = solve(M, z)
    w = np.append(np.append([0], s), [0])

    # initializing a,b,c,d
    a = np.array([0.0] * (n + 1))
    b = np.array([0.0] * (n + 1))
    c = np.array([0.0] * (n + 1))
    d = np.array([0.0] * (n + 1))
    q = np.array([0.0] * (n + 1))
    r = np.array([0.0] * (n + 1))

    for i in range(1, n + 1):
        c[i] = (w[i - 1] * x[i] - w[i] * x[i - 1]) / (2 * (x[i] - x[i - 1]))
        d[i] = (w[i] - w[i - 1]) / (6 * (x[i] - x[i - 1]))

    for i in range(1, n + 1):
        q[i - 1] = v[i - 1] - (c[i] * x[i - 1] ** 2) - (d[i] * x[i - 1] ** 3)
        r[i] = v[i] - (c[i] * x[i] ** 2) - (d[i] * x[i] ** 3)

    for i in range(1, n + 1):
        a[i] = (q[i - 1] * x[i] - r[i] * x[i - 1]) / (x[i] - x[i - 1])
        b[i] = (r[i] - q[i - 1]) / (x[i] - x[i - 1])

    return a[1:], b[1:], c[1:], d[1:], M, z


def cubic_x(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3


def piecewise_cubic(x, x_list, a, b, c, d):
    assert len(x_list) == len(a) + 1
    assert len(x_list) == len(b) + 1
    assert len(x_list) == len(c) + 1
    assert len(x_list) == len(d) + 1

    for i in range(1, len(x_list)):
        if x <= x_list[i]:
            return cubic_x(x, a[i-1], b[i-1], c[i-1], d[i-1])


def rate_curve(t, t_list, a, b, c, d):
    for i in range(1, len(t_list)):
        if t <= t_list[i]:
            return cubic_x(t, a[i-1], b[i-1], c[i-1], d[i-1])


""" --- NLA Methods --- """


def min_var_port(cov_mat, mu, r, mu_req):
    mu_bar = mu - r
    wt = tangency_port(cov_mat, mu_bar)
    wcash = 1 - (mu_req - r) / np.matmul(mu_bar.transpose(), wt)
    w = (1 - wcash) * wt
    sigma = np.sqrt(np.matmul(np.matmul(w.transpose(), cov_mat), w))
    return w, wcash, sigma


def min_var_overall_port(cov_mat):
    ones = np.array([1] * len(cov_mat)).reshape(-1, 1)
    Ut = np.linalg.cholesky(cov_mat)
    x = cho_solve((Ut, True), ones)

    return x / ones.transpose().dot(x)


def max_ret_port(cov_mat, mu, r, sig_req):
    mu_bar = mu - r
    Ut = np.linalg.cholesky(cov_mat)
    x = cho_solve((Ut, True), mu_bar)
    ones = np.array([1] * len(x)).reshape(-1, 1)
    wt = (1 / np.matmul(ones.transpose(), x)) * x
    if ones.transpose().dot(x) > 0:
        wcash = 1 - (sig_req / np.sqrt(wt.transpose().dot(cov_mat).dot(wt)))
    else:
        wcash = 1 + (sig_req / np.sqrt(wt.transpose().dot(cov_mat).dot(wt)))
    w = (1 - wcash) * wt
    mu_max = r + mu_bar.transpose().dot(w)
    return w, wcash, mu_max


def tangency_port(cov_mat, mu_bar):
    Ut = np.linalg.cholesky(cov_mat)
    x = cho_solve((Ut, True), mu_bar)
    ones = np.array([1] * len(x)).reshape(-1, 1)
    return (1 / np.matmul(ones.transpose(), x)) * x


def fully_invested_port(cov_mat):
    ones = np.array([1] * cov_mat.shape[0]).reshape(-1,1)
    Ut = np.linalg.cholesky(cov_mat)
    x = cho_solve((Ut, True), ones)
    return (1 / np.matmul(ones.transpose(), x)) * x