from sympy.solvers import solve
from sympy import Symbol
import numpy as np
from scipy.stats.mstats import gmean
import pandas as pd
import matplotlib.pyplot as plt

sign = lambda a: int((a > 0)) - int((a < 0))


def getline(ws, xu, yu, xl, yl):
    x = Symbol('x')
    y = Symbol('y')
    epsi = 10 ** (-12)

    if abs(ws[0]) >= epsi and abs(ws[1]) >= epsi:
        try:
            xxu = float(solve(ws[0] * x + ws[1] * yu + ws[2], x)[0].evalf())
            xxl = float(solve(ws[0] * x + ws[1] * yl + ws[2], x)[0].evalf())
            yyl = float(solve(ws[0] * xl + ws[1] * y + ws[2], y)[0].evalf())
            yyu = float(solve(ws[0] * xu + ws[1] * y + ws[2], y)[0].evalf())
        except:
            xxu = (-1) * (ws[1] * yu + ws[2]) / ws[0]
            xxl = (-1) * (ws[1] * yl + ws[2]) / ws[0]
            yyl = (-1) * (ws[0] * xl + ws[2]) / ws[1]
            yyu = (-1) * (ws[0] * xu + ws[2]) / ws[1]

        candidates = [(xxu, yu), (xxl, yl), (xl, yyl), (xu, yyu)]
        tuples = []
        for tup in candidates:
            if xu >= tup[0] >= xl and yu >= tup[1] >= yl:
                tuples.append(tup)

    elif abs(ws[0]) >= epsi > abs(ws[1]):  # y coeff is small
        try:
            xx = float(solve(ws[0] * x + ws[2], x)[0].evalf())
        except:
            xx = (-1) * ws[2] / ws[0]

        if xu >= xx >= xl:
            tuples = [(xx, yl), (xx, yu)]
        elif xx > xu:
            tuples = [(xu, yl), (xu, yu)]
        elif xx < xl:
            tuples = [(xl, yl), (xl, yu)]

    elif abs(ws[0]) < epsi <= abs(ws[1]):  # x coeff is small
        try:
            yy = float(solve(ws[1] * y + ws[2], y)[0].evalf())
        except:
            yy = (-1) * ws[2] / ws[1]

        if yy <= yu and yy >= yl:
            tuples = [(xu, yy), (xl, yy)]
        elif yy > yu:
            tuples = [(xu, yu), (xl, yu)]
        elif yy < yl:
            tuples = [(xl, yl), (xl, yl)]

    else:
        print("No Valid Separating Hyperplanes")
        return None
    return tuples


def print_MIP(obj_inf, exact_obj_inf, feasible, name=""):
    """
    print the MIP info
    :param obj_inf:
    :param exact_obj_inf:
    :param feasible:
    :param name:
    :return:
    """
    exact_obj = exact_obj_inf.Risk + exact_obj_inf.Margin
    obj = obj_inf.Risk + obj_inf.Margin
    MIP_Gaps = (obj[feasible] / exact_obj[feasible] - 1)
    # MIP_Gaps.boxplot()
    #
    if name != "":
        print(name)
    print("Median MIP Gap: ", np.percentile(MIP_Gaps, 50))
    print("95% MIP Gap: ", np.percentile(MIP_Gaps, 95))
    print("70% MIP Gap: ", np.percentile(MIP_Gaps, 70))
    print("5% MIP Gap: ", np.percentile(MIP_Gaps, 5))
    return MIP_Gaps


def max_dd(returns):
    """Assumes returns is a pandas Series"""
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.argmin()
    start = r.iloc[:end].argmax()
    return mdd, start, end


def RetStats(returns, rf=None, geomean=True):
    """Assumes returns is a pandas Series"""
    if rf is None:
        returns_rf = returns
    else:
        returns_rf = returns - rf
    P = 12
    P05 = P ** 0.5

    if geomean:
        mean = gmean(1 + returns_rf) - 1
    else:
        mean = returns_rf.mean()
    return P * mean, P05 * returns_rf.std(), P * mean / (P05 * returns_rf.std())


def wealth(returns):
    """Assumes returns is a pandas Series"""
    return (1 + returns).cumprod()


def load_data_mth(rets, forecasts, wrds_svm, cols, prd, N, data_treatment=None):
    ret_ = rets.loc[prd][:N]
    cov_, mean_ = forecasts[prd]
    cov = cov_[:N, :N]
    mean = mean_[:N, :]
    tics = list(rets.columns[:N])
    # get the wharton research data for the valid tickers for the month
    wrds_tics = wrds_svm[wrds_svm.index.get_level_values('tic').isin(tics)].xs(prd, level="MonthStart")
    # restrict the wharton research data to the columns of interest
    # modifying WRDS dataset here if required
    Y = wrds_tics.loc[:, cols]  # Y matrix in formulation

    if data_treatment:
        Y_out = Y
    else:
        Y_out = data_treatment(Y)
    return ret_, cov, mean, tics, wrds_tics, Y_out


def evaluate_model(rets, forecasts, wrds_svm, return_premium, model_instance, T, N, cols, turnover_limit,
                   cbb_fn=None, lr=None, data_treatment=None):
    """
    Runs the experiment on the model_instance
    :param rets:
    :param forecasts:
    :param wrds_svm:
    :param return_premium:
    :param model_instance:
    :param T:
    :param N:
    :param cols:
    :param turnover_limit:
    :param cbb_fn:
    :param lr:
    :return:
    """
    weights = np.zeros([T, N])
    oot_returns = np.zeros(T)
    market = np.zeros(T)
    M = len(cols)
    wis = np.zeros([T, M])
    bias = []
    i = 0  # index for dates
    w_mabs = 0  # initialize
    soln_mods = {}

    for prd in rets.index.to_list()[:T]:

        ret_, cov, mean, tics, wrds_tics, Y_out = load_data_mth(rets,
                                                                forecasts,
                                                                wrds_svm,
                                                                cols,
                                                                prd,
                                                                N,
                                                                data_treatment=data_treatment)

        return_premium_temp = return_premium
        ret_constr = mean.mean() * (1 + sign(mean.mean()) * return_premium)
        # update model
        model_instance.tics = tics
        model_instance.ret_constr = ret_constr
        model_instance.mean_ret = mean
        model_instance.cov = cov
        model_instance.exogenous = Y_out


        warm_starts = None
        if i > 0:  # not the first trade gets a constraint on turnover
            x_prev = model_instance.x.X
            if model_instance.svm_constr:
                warm_starts = [model_instance.x.x, model_instance.z.x, model_instance.w.x, model_instance.b.x]
            if model_instance.svm_constr:
                w_mabs = (i / (i + 1)) * w_mabs + (1 / (i + 1)) * np.abs(model_instance.w.x).mean()
                w_prev = model_instance.w.x

        model_instance.set_model(start=warm_starts)

        if i > 0:  # not the first trade gets a constraint on turnover
            model_instance.define_turnover(x_prev, np.ones_like(x_prev), turnover_limit, 1)

            if model_instance.svm_constr and lr is not None:
                wcon1 = model_instance.model.addConstr(model_instance.w <= w_prev + lr * w_mabs,
                                                       'iter constraint 1')
                wcon2 = model_instance.model.addConstr(model_instance.w >= w_prev - lr * w_mabs,
                                                       'iter constraint 2')

        model_instance.model.Params.LogToConsole = 0

        model_instance.optimize(cbb=cbb_fn)
        k = 1
        while model_instance.model.status == 4:
            # if the model is infeasible the decrease the return constraint
            # we do not have enough turnover the modify the portfolio to achive the
            # return target... not a great place to be
            return_premium_temp = return_premium - 0.05 * k
            ret_constr = mean.mean() * (1 + sign(mean.mean()) * return_premium_temp)
            model_instance.ret_constr = ret_constr
            model_instance.ret_target[0].rhs = ret_constr
            model_instance.optimize(cbb=cbb_fn)
            k = k + 1
            # relax the return target bc the turnover constraint isnt allowing it to solve
        # oot_returns[i] = model_instance.evaluate(ret_)
        market[i] = ret_.mean()
        weights[i, :] = model_instance.x.x
        if model_instance.model.IsMIP:
            soln_mods[prd] = [model_instance.model.MIPGap, return_premium_temp]
        else:
            soln_mods[prd] = return_premium_temp
        if model_instance.svm_constr:
            wis[i, :] = model_instance.w.x
            bias.append(model_instance.b.x)
            # if model_instance.w.x >= w_prev + lr*w_mabs  and wcon2.Pi < 10**(-7):
            #   lr = lr/2
        if i + 1 >= T:
            break
        if i % 12 == 0:
            print("_" * 25)
            print("Iteration ", i)
            print("Percent Complete ", i / T)
        i = i + 1

    weights = pd.DataFrame(weights, index=rets.index[:T], columns=model_instance.tics)
    oot_returns = pd.Series(oot_returns, index=rets.index[:T])
    market = pd.Series(market, index=rets.index[:T])
    return weights, oot_returns, market, wis, bias, soln_mods


def evaluate_adm(rets, forecasts, wrds_svm, return_premium, model_instance, T, N, cols,
                 turnover_limit, lr=None, data_treatment=None):
    portfolio_weights = np.zeros([T, N])
    oot_returns = np.zeros(T)
    market = np.zeros(T)
    M = len(cols)
    wis = np.zeros([T, M])
    bias = np.zeros(T)
    times = []
    i = 0  # index for dates
    w_mabs = 10 ** (-9)  # initialize
    soln_mods = {}

    for prd in rets.index.to_list()[:T]:

        ret_, cov, mean, tics, wrds_tics, Y_out = load_data_mth(rets, forecasts, wrds_svm,
                                                                cols, prd, N, data_treatment=data_treatment)
        return_premium_temp = return_premium
        if return_premium == -1:
            ret_constr = -1
        else:
            ret_constr = mean.mean() * (1 + sign(mean.mean()) * return_premium)
        model_instance.MVO_.tics = tics
        model_instance.SVM_.tics = tics
        model_instance.MVO_.ret_constr = ret_constr
        model_instance.MVO_.mean_ret = mean
        model_instance.MVO_.cov = cov
        model_instance.MVO_.exogenous = Y_out
        model_instance.SVM_.exogenous = Y_out
        mvo_cons = []
        svm_cons = []
        w_prev = None
        if i > 0:  # not the first trade gets a constraint on turnover
            # turnover constraint
            x_prev = model_instance.MVO_.x.X

            warm_starts = [model_instance.MVO_.x.X, model_instance.MVO_.z.X]

            # policy constraint
            w_mabs = (i / (i + 1)) * w_mabs + (1 / (i + 1)) * np.abs(model_instance.SVM_.w.x).mean()


            # portfolio turnover constraints
            for v, absv, curr in zip(model_instance.MVO_.x.tolist(), model_instance.MVO_.abs.tolist(), x_prev.tolist()):
                mvo_cons.append(absv >= v - curr)
                mvo_cons.append(absv >= curr - v)
                # q = cost*1/np.maximum(1, Prices)
            q = 1 * 1 / np.maximum(1, np.ones_like(x_prev))
            mvo_cons.append(model_instance.MVO_.abs @ q <= turnover_limit)
            # add constraints on w

            # epsilon allows change if w is 0
            if lr is not None:
                w_prev = model_instance.SVM_.w.x
                b_prev = model_instance.SVM_.b.x
                # svm_cons.append(
                #     model_instance.SVM_.w <= w_prev + lr * w_mabs)  # 'iter constraint 1'
                # svm_cons.append(
                #     model_instance.SVM_.w >= w_prev - lr * w_mabs)  # 'iter constraint 2'

        try:
            model_instance.initialize_soln(constrs=mvo_cons, svm_constrs=svm_cons,
                                           warm_starts=warm_starts, delta=lr, w_prev_soln=w_prev)
        except:
            print("Begin Relaxation")
        k = 1
        while model_instance.MVO_.model.status == 4:
            # if the model is infeasible the decrease the return constraint
            # we do not have enough turnover the modify the portfolio to achive the
            # return target... not a great place to be
            print("Infeasible return constraint...Relaxing")
            return_premium_temp = return_premium - 0.05 * k
            ret_constr = mean.mean() * (1 + sign(mean.mean()) * return_premium_temp)
            model_instance.MVO_.ret_constr = ret_constr
            model_instance.MVO_.ret_target[0].rhs = ret_constr
            try:
                model_instance.initialize_soln(constrs=mvo_cons, svm_constrs=svm_cons,
                                               warm_starts=warm_starts, delta=lr, w_prev_soln=w_prev)
            except:
                print("Try to Relax Again")
            k = k + 1

        # model_instance.silence_output()
        try:
            ws, xs, zs, xi_mvo, xi_svm, dt, objectives_svm, objectives_mvo, penalty_hist = \
                model_instance.solve_adm(store_data=False, constrs=mvo_cons, svm_constrs=svm_cons,
                                         delta=lr, w_prev_soln=w_prev)
        except:
            print("Begin Relaxation")
        k = 1
        while model_instance.MVO_.model.status == 4:
            # if the model is infeasible the decrease the return constraint
            # we do not have enough turnover the modify the portfolio to achive the
            # return target... not a great place to be
            print("Infeasible return constraint...Relaxing")
            return_premium_temp = return_premium - 0.05 * k
            ret_constr = mean.mean() * (1 + sign(mean.mean()) * return_premium_temp)
            model_instance.MVO_.ret_constr = ret_constr
            model_instance.MVO_.ret_target[0].rhs = ret_constr
            try:
                ws, xs, zs, xi_mvo, xi_svm, dt, objectives_svm, objectives_mvo, penalty_hist  = \
                    model_instance.solve_adm(store_data=False, constrs=mvo_cons, svm_constrs=svm_cons,
                                             delta=lr, w_prev_soln=w_prev)
            except:
                print("Try to Relax Again")
            if k > 3: #only try to relax a couple times
                model_instance.MVO_.ret_constr = -1
                model_instance.MVO_.ret_target[0].rhs = -1
                print("giving up ...  MVP")
                ws, xs, zs, xi_mvo, xi_svm, dt, objectives_svm, objectives_mvo, penalty_hist  = \
                    model_instance.solve_adm(store_data=False, constrs=mvo_cons, svm_constrs=svm_cons,
                                             delta=lr, w_prev_soln=w_prev)
                break
            k = k + 1

        # if model_instance.model.IsMIP:
        #   soln_mods[prd] = [model_instance.model.MIPGap, return_premium_temp]
        # else:
        model_instance.w = model_instance.w.x  # convert to numpy array
        model_instance.b = model_instance.b.x
        # alpha = 0.95
        # if i > 0:
        #   model_instance.MVO_.svm_w = alpha*(model_instance.SVM_.w.x) + (1- alpha)*(w_prev)
        #   model_instance.MVO_.svm_b = alpha*(model_instance.SVM_.b.x) + (1- alpha)*(b_prev)
        #   model_instance.MVO_.optimize()
        #   model_instance.x =  model_instance.MVO_.x
        #   model_instance.w =  model_instance.MVO_.svm_w #numpy
        #   model_instance.b =  model_instance.MVO_.svm_b #numpy

        soln_mods[prd] = return_premium_temp
        # oot_returns[i] = model_instance.evaluate(ret_)
        market[i] = ret_.mean()
        portfolio_weights[i, :] = model_instance.x.x
        times.append(dt)
        wis[i, :] = model_instance.w
        bias[i] = model_instance.b

        if i + 1 >= T:
            break

        if i % 12 == 0:
            print("_" * 25)
            print("Iteration ", i)
            print("Percent Complete ", i / T)
            print(model_instance.w)
        i = i + 1

    portfolio_weights = pd.DataFrame(portfolio_weights, index=rets.index[:T], columns=model_instance.tics)
    oot_returns = pd.Series(oot_returns, index=rets.index[:T])
    market = pd.Series(market, index=rets.index[:T])
    return (portfolio_weights, oot_returns, market, wis, bias, soln_mods, times)
