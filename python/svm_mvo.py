import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

pth = ''

e = gp.Env(empty=True)
# e.setParam('OutputFlag', 0)
# gurobi_licence = pd.read_csv(pth +'SVM MVO/cache/gurobi.csv')
# e.setParam('WLSACCESSID', gurobi_licence.WLSACCESSID[0])
# e.setParam('LICENSEID', gurobi_licence.LICENSEID[0])
# e.setParam('WLSSECRET', gurobi_licence.WLSSECRET[0])
e.start()


def portfolio_risk_posdef(optimizer):
    """

    :param optimizer:
    :return:
    """
    min_eig = np.min(np.linalg.eigvals(optimizer.cov)) - 10 ** (-8)
    if min_eig < 0:
        raise ValueError('Negative eigenvalues')
    return optimizer.v.sum() + optimizer.q @ (optimizer.cov - optimizer.posdef_diag) @ optimizer.q


# noinspection PyTypeChecker
class SVMMVO:
    big_m = 100

    def __init__(self, tics, mean_ret, cov, ret_constr, soft_margin, exogenous, asset_lim,
                 svm_choice=(False, False), print_var_frntr=False, indicator=False,
                 cardinality=False, non_neg=True, lower_asset_lim=0, epsilon=0.001):
        self.tics = tics  # list of tickers
        self.mean_ret = mean_ret
        self.cov = cov
        self.ret_constr = ret_constr
        self.AssetLim = asset_lim
        self.AssetLim_Lower = lower_asset_lim
        self.soft_margin = soft_margin  # hyper parameter
        self.exogenous = exogenous  # matrix of features for the tickers
        self.svm_constr, self.slacks = svm_choice  # Model configuration
        self.indicator = indicator
        self.cardinality = cardinality
        self.epsilon = epsilon

        n, m = self.exogenous.shape

        self.model = gp.Model(env=e)
        self.x = self.model.addMVar(n)
        if non_neg:
            self.w = self.model.addMVar(m)
            self.b = self.model.addMVar(1)
        else:
            self.w = self.model.addMVar(m, lb=-1 * np.inf)
            self.b = self.model.addMVar(1, lb=-1 * np.inf)
        self.z = self.model.addMVar(n, vtype=GRB.BINARY)
        self.xi = self.model.addMVar(n, lb=np.zeros(n))
        self.v = self.model.addMVar(n, lb=np.zeros(n))
        self.q = self.model.addMVar(n)

        self.print_var_frntr = print_var_frntr
        self.ret_target = self.model.addConstr(self.port_exptd_ret >= self.ret_constr, 'target')
        self.model.update()

        self.abs = self.model.addMVar(n)
        min_eig = min(np.linalg.eigvals(self.cov)) - 10 ** (-8)
        self.posdef_diag = np.diag(np.ones(n)) * min_eig

    @property
    def portfolio_risk(self):
        return self.x @ self.cov @ self.x

    @property
    def port_exptd_ret(self):
        return self.mean_ret[:, 0] @ self.x

    @property
    def soft_penalty(self):
        n, m = self.exogenous.shape
        return (self.soft_margin / n) * self.xi.sum()

    @property
    def svm_margin(self):
        return (1 / 2) * (self.w @ self.w)

    @property
    def portfolio_risk_p(self):
        return portfolio_risk_posdef(self)

    @property
    def describe(self):
        ret_pct = round(100 * self.ret_constr)
        if self.ret_constr == -1:
            if self.svm_constr:
                if self.slacks:
                    desc = "SVM MVO with Slacks Min Variance Portfolio"
                    shrt = "SVMMVO_Slck_ret" + str(ret_pct) + "%"
                else:
                    desc = "SVM MVO with no Slacks Min Variance Portfolio"
                    shrt = "SVMMVO_ret" + str(ret_pct) + "%"
            else:
                desc = "Traditional MVO Min Variance Portfolio"
                shrt = "MVO_ret" + str(ret_pct) + "%"
        else:
            if self.svm_constr:
                if self.slacks:
                    desc = "SVM MVO with Slacks with return exceeding " + str(ret_pct) + "%"
                    shrt = "SVMMVO_Slck_ret" + str(ret_pct) + "%"
                else:
                    desc = "SVM MVO with no Slacks with return = " + str(ret_pct) + "%"
                    shrt = "SVMMVO_ret" + str(ret_pct) + "%"
            else:
                desc = "Traditional MVO with return exceeding " + str(ret_pct) + "%"
                shrt = "MVO_ret" + str(ret_pct) + "%"
        return desc, shrt

    def print_var_info(self, names=None):
        # dictionary of variables and thier names
        if names is None:
            names = {}
        for key in names.keys():
            print(key, names[key])
        print("x", self.x.X)
        print("w", self.w.X)
        print("z", self.z.X)
        print("xi", self.xi.X)
        print("")

    def set_model(self, start=None):
        # remove constraints
        if start is None:
            start = []

        self.model.remove(self.model.getConstrs())
        # parameter definitions
        n, m = self.exogenous.shape
        big_m: int = SVMMVO.big_m
        epsilon = self.epsilon
        # objective function components

        # user supplied start values
        if start:  # start is a list
            self.x.start = start[0]
            self.z.start = start[1]
            self.w.start = start[2]
            self.b.start = start[3]

        self.model.remove(self.ret_target)
        self.ret_target = self.model.addConstr(self.port_exptd_ret >= self.ret_constr, 'target')

        self.model.update()
        self.model.addConstr(self.x.sum() == 1, 'budget')

        if not self.svm_constr and self.slacks:
            # if we do not want to use svm then don't use svm with slack
            print("Cannot SVM with slack without SVM ... self.slacks set to False")
            self.slacks = False

        if self.svm_constr:  # SVM type Models
            self.model.addConstr(self.x <= self.z, "z force x")  # if x is close to zero then z must be zero 
            self.model.addConstr(self.z.sum() <= self.AssetLim, "cardinality 1")
            self.model.addConstr(self.z.sum() >= self.AssetLim_Lower, "cardinality lower")
            # if self.slacks and not self.indicator: #SVM MVO with slack variables
            if self.slacks is True and self.indicator is False:  # SVM MVO with slack variables

                self.model.setObjective(self.portfolio_risk + self.svm_margin + self.soft_penalty, GRB.MINIMIZE)
                self.model.remove([self.v, self.q])
                # adding the SVM type constraints
                for i in range(n):
                    y_i = self.exogenous.iloc[i].values
                    self.model.addConstr(self.w @ y_i + self.b <= (-1) * self.epsilon + self.xi[i] + big_m * self.z[i],
                                         "svm1")
                    self.model.addConstr(-1 * big_m * (1 - self.z[i]) + self.epsilon - self.xi[i]
                                         - (y_i @ self.w + self.b) <= 0,
                                         "svm2")

            elif self.slacks is True and self.indicator is True:
                # SVM MVO with slack variables with indicator type constraints

                self.model.setObjective(self.portfolio_risk + self.svm_margin + self.soft_penalty, GRB.MINIMIZE)
                self.model.remove([self.v, self.q])
                # adding the SVM type constraints
                for i in range(n):
                    y_i = self.exogenous.iloc[i].values
                    w_ = self.w.tolist()
                    xi_ = self.xi.tolist()
                    b_ = self.b.tolist()
                    z_ = self.z.tolist()
                    self.model.addGenConstrIndicator(z_[i], True,
                                                     lhs=gp.quicksum(w_[j] * y_i[j] for j in range(m)) + b_[0] + xi_[i],
                                                     sense=GRB.GREATER_EQUAL, rhs=1.0 * self.epsilon,
                                                     name="svm1")
                    self.model.addGenConstrIndicator(z_[i], False,
                                                     lhs=gp.quicksum(w_[j] * y_i[j] for j in range(m)) + b_[0] - xi_[i],
                                                     sense=GRB.LESS_EQUAL,
                                                     rhs=(-1.0) * self.epsilon, name="svm2")

            else:  # SVM MVO without slack variables - may not be feasible
                self.model.remove([self.xi, self.v, self.q])
                self.model.setObjective(self.portfolio_risk + self.svm_margin, GRB.MINIMIZE)
                # adding the SVM type constraints
                for i in range(n):
                    y_i = self.exogenous.iloc[i].values
                    self.model.addConstr(self.w @ y_i + self.b <= (-1) * self.epsilon + big_m * self.z[i], "svm1")
                    self.model.addConstr(-1*big_m * (1 - self.z[i]) + 1*self.epsilon
                                         - (y_i @ self.w + self.b) <= 0, "svm2")
        elif self.cardinality:
            self.model.update()
            self.model.addConstr(self.z.sum() <= self.AssetLim, 'Cardinality')
            self.model.addConstr(self.z.sum() >= self.AssetLim_Lower, 'Cardinality')
            z_ = self.z.tolist()
            x_ = self.x.tolist()
            self.model.addConstr(self.x <= self.z, name="z force x")
            self.model.setObjective(self.portfolio_risk, GRB.MINIMIZE)

            for i in range(n):
                self.model.addGenConstrIndicator(z_[i], True, lhs=x_[i], sense=GRB.GREATER_EQUAL,
                                                 rhs=self.epsilon / big_m,
                                                 name="indicator constraint")
            self.model.remove([self.w, self.b, self.xi, self.v, self.q])

        else:  # not svm mvo
            self.model.setObjective(self.portfolio_risk, GRB.MINIMIZE)
            self.model.remove([self.w, self.b, self.z, self.xi, self.v, self.q])

    def optimize(self, cbb=None):

        if cbb not in [None]:
            self.model._cur_obj = float('inf')
            self.model._time = time.time()
            start = time.time()
            self.model.optimize(callback=cbb)
            end = time.time()
        else:
            start = time.time()
            self.model.optimize()
            end = time.time()
        self.model.write('portfolio_selection_optimization.lp')
        return end - start

    def evaluate(self, realized_ret):

        ret = np.dot(self.x.X, realized_ret)
        return ret  # use this to calculate out of sample rets and var

    def get_estimates(self):

        vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
        expt_ret_metric = self.port_exptd_ret.getValue()[0]
        return [vol_metric, expt_ret_metric]  # use this for efficient frontiers

    def get_results(self, export_dir='', fig_size=()):

        lng, shrt = self.describe
        vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
        expt_ret_metric = self.port_exptd_ret.getValue()[0]
        results = pd.DataFrame(data=np.append(self.x.X, [vol_metric, expt_ret_metric]),
                               index=list(self.tics) + ['Volatility', 'Expected Return'], columns=[lng])

        if export_dir != '':
            results.to_csv(export_dir + 'results.csv')

        if fig_size != () and type(fig_size) in [list, tuple]:
            results[:-2].plot.bar(figsize=fig_size)

        return results.transpose()

    def get_frontier(self, export_dir='', fig_size=(10, 8), lower_ret=None, upper_ret=None):

        n, m = self.exogenous.shape
        f = 25
        mean_ret = self.mean_ret[:, 0]
        # F is the number of portfolios to use for frontier
        frontier = np.empty((2, f))
        # ws will contain the w's and b values for each portfolio 
        ws = np.empty((f, m + 1))
        # xis will contain the w's and b values for each portfolio 
        xis = np.empty((f, n))
        # targets for returns
        if lower_ret is None:
            lower_ret = mean_ret.min()
        if upper_ret is None:
            upper_ret = mean_ret.max()
        ret_targ = np.linspace(lower_ret, upper_ret, f)

        expt_ret_metric = self.port_exptd_ret.getValue()[0]

        self.model.remove(self.ret_target)
        self.model.update()

        self.ret_target = self.model.addConstr(self.port_exptd_ret == expt_ret_metric, 'target ==')
        self.model.update()

        for i in range(f):
            self.ret_target[0].rhs = ret_targ[i]
            self.model.optimize()

            if self.model.status == 4:
                break

            vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
            frontier[:, i] = np.array([vol_metric, ret_targ[i]])

            if self.svm_constr and self.slacks:
                ws[i, :] = np.concatenate([self.w.x, self.b.x])
                xis[i, :] = self.xi.x
                if self.print_var_frntr:
                    self.print_var_info({"return": ret_targ[i]})

        if self.model.status == 4:
            print("Resolving Model to initial state (return target) then exiting")
            self.model.remove(self.ret_target)
            self.model.update()
            self.ret_target = self.model.addConstr(self.port_exptd_ret >= self.ret_constr, 'target')
            self.model.optimize()
            return None, None, None

        # restore model to original state
        self.model.remove(self.ret_target)
        self.model.update()
        self.ret_target = self.model.addConstr(self.port_exptd_ret >= self.ret_constr, 'target')
        self.model.optimize()

        fig, ax = plt.subplots(figsize=fig_size)
        # Plot efficient frontier
        ax.plot(frontier[0], frontier[1], '-*', label='Efficient Frontier', color='DarkGreen')

        # Format and display the final plot
        ax.axis([frontier[0].min() * 0.7, frontier[0].max() * 1.3, mean_ret.min() * 1.2, mean_ret.max() * 1.2])
        ax.set_xlabel('Volatility (standard deviation)')
        ax.set_ylabel('Expected Return')
        # ax.legend()
        ax.grid()
        plt.show()
        if export_dir != '':
            plt.savefig(export_dir + "EfficientFrontier.png")
        return frontier, ws, xis

    def define_turnover(self, x0, prices, limit, cost):
        for v, absv, curr in zip(self.x.tolist(), self.abs.tolist(), x0.tolist()):
            self.model.addConstr(absv >= v - curr, "turnover constraint1")
            self.model.addConstr(absv >= curr - v, "turnover constraint2")
        q = cost * 1 / np.maximum(1, prices)
        self.model.addConstr(self.abs @ q <= limit, "turnover constraint3")


class MVO:
    # This class models the mean variance sub-problem in the ADM method

    big_m = 100

    # noinspection PyTypeChecker
    def __init__(self, tics, mean_ret, cov, ret_constr, exogenous, asset_lim,
                 soft_margin=0, svm_w=None, svm_b=None,
                 perspective=False, asset_lim_lower=0, epsilon=0.001):
        self.tics = tics  # list of tickers
        self.mean_ret = mean_ret
        self.cov = cov
        self.ret_constr = ret_constr
        self.AssetLim = asset_lim
        self.AssetLim_Lower = asset_lim_lower
        self.exogenous = exogenous  # matrix of features for the tickers
        n, m = self.exogenous.shape

        self.model = gp.Model(env=e)
        self.x = self.model.addMVar(n)
        self.z = self.model.addMVar(n, vtype=GRB.BINARY)
        self.v = self.model.addMVar(n, lb=np.zeros(n))
        self.q = self.model.addMVar(n)
        self.xi = self.model.addMVar(n, lb=np.zeros(n))
        self.ret_target = self.model.addConstr(self.port_exptd_ret >= self.ret_constr, 'target')

        self.svm_w = svm_w
        self.svm_b = svm_b
        self.soft_margin = soft_margin
        self.perspective = perspective
        self.epsilon = epsilon
        self.model.update()

        self.abs = self.model.addMVar(n)

        min_eig = min(np.linalg.eigvals(self.cov)) - 10 ** (-8)
        self.posdef_diag = np.diag(np.ones(n)) * min_eig

    @property
    def portfolio_risk(self):
        return self.x @ self.cov @ self.x

    @property
    def port_exptd_ret(self):
        return self.mean_ret[:, 0] @ self.x

    @property
    def soft_penalty(self):
        n, m = self.exogenous.shape
        if np.isscalar(self.soft_margin):
            return (1 / n) * self.soft_margin * (self.xi.sum())
        else:
            return (1 / n) * (self.soft_margin[:, 0] @ self.xi)

    @property
    def portfolio_risk_p(self):
        return portfolio_risk_posdef(self)

    def print_var_info(self, names=None):

        # dictionary of variables and their names
        if names is None:
            names = {}
        for key in names.keys():
            print(key, names[key])
        print("x", self.x.X)
        print("z", self.z.X)
        print("")

    def set_model(self, set_return=True, constrs=None, warm_starts=None):

        # parameter definitions
        if constrs is None:
            constrs = []
        if warm_starts is None:
            warm_starts = []

        if warm_starts:  # start is a list
            self.x.start = warm_starts[0]
            self.z.start = warm_starts[1]
        self.model.remove(self.model.getConstrs())
        n, m = self.exogenous.shape
        big_m = MVO.big_m
        epsilon = self.epsilon
        # objective function components
        if set_return:
            # remove constraints and reset the return constraints
            self.ret_target = self.model.addConstr(self.port_exptd_ret >= self.ret_constr, 'target')
        if constrs:
            for con in constrs:
                self.model.addConstr(con, 'target')

        self.model.update()
        self.model.addConstr(self.x.sum() == 1, 'budget')
        self.model.addConstr(self.z.sum() <= self.AssetLim, 'Cardinality')
        self.model.addConstr(self.z.sum() >= self.AssetLim_Lower, 'Cardinality')

        self.model.addConstr(self.x <= self.z, "z force x")
        self.model.setObjective(self.portfolio_risk + self.soft_penalty, GRB.MINIMIZE)

        # for i in range(N):
        #   self.model.addGenConstrIndicator(z_[i], True, x_[i] >= self.epsilon/big_m)
        # self.model.addGenConstrIndicator(z_[i], False, x_[i] <= self.epsilon/(100*big_m) - 10**(-7))

        # the SVM info must be uninitialized on the first run
        if type(self.svm_w) is np.ndarray:
            for i in range(n):
                y_i = self.exogenous.iloc[i].values
                a = np.dot(y_i, self.svm_w) + self.svm_b
                self.model.addConstr((-1) * self.epsilon + self.xi[i] + big_m * self.z[i] >= a, "svm1")
                self.model.addConstr(-1 * big_m * (1 - self.z[i]) + 1 * self.epsilon - self.xi[i] - a <= 0, "svm2")
                # self.model.addConstr(2*a[0]*z_[i] >= a[0] + self.epsilon - xi_[i], 'svm'+str(i))

    def optimize(self, cbb=None):

        if cbb not in [None]:
            self.model._cur_obj = float('inf')
            self.model._time = time.time()
            self.model.optimize(callback=cbb)
        else:
            self.model.optimize()
        self.model.write('portfolio_selection_optimization.lp')

    def evaluate(self, realized_ret):

        ret = np.dot(self.x.X, realized_ret)
        return ret  # use this to calculate out of sample rets and var

    def get_estimates(self):

        vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
        expt_ret_metric = self.port_exptd_ret.getValue()[0]
        return [vol_metric, expt_ret_metric]  # use this for efficient frontiers

    def define_turnover(self, x0, prices, limit, cost):
        for v, absv, curr in zip(self.x.tolist(), self.abs.tolist(), x0.tolist()):
            self.model.addConstr(absv >= v - curr, "turnover constraint1")
            self.model.addConstr(absv >= curr - v, "turnover constraint2")
        q = cost * 1 / np.maximum(1, prices)
        self.model.addConstr(self.abs @ q <= limit, "turnover constraint3")


class SVM:
    # This class models the support vector machine subproblem in the ADM method

    big_m = 100

    # noinspection PyTypeChecker
    def __init__(self, tics, exogenous, soft_margin, mvo_z=None, non_neg=True, epsilon=0.001):
        self.tics = tics  # list of tickers
        self.exogenous = exogenous  # matrix of features for the tickers
        self.soft_margin = soft_margin  # hyper parameter

        n, m = self.exogenous.shape

        self.model = gp.Model(env=e)

        if non_neg:
            self.w = self.model.addMVar(m)
            self.b = self.model.addMVar(1)
        else:
            self.w = self.model.addMVar(m, lb=-1 * np.inf)
            self.b = self.model.addMVar(1, lb=-1 * np.inf)
        self.xi = self.model.addMVar(n, lb=np.zeros(n))

        self.mvo_z = mvo_z
        self.abs_b = self.model.addMVar(1)
        self.abs = self.model.addMVar(n)
        self.epsilon = epsilon

    @property
    def soft_penalty(self):
        n, m = self.exogenous.shape
        if np.isscalar(self.soft_margin):
            return (1 / n) * self.soft_margin * (self.xi.sum())
        else:

            return (1 / n) * (self.soft_margin[:, 0] @ self.xi)

    @property
    def svm_margin(self):
        return (1 / 2) * (self.w @ self.w)

    def svm_change(self, w_prev):
        n, m = self.exogenous.shape
        w_diff = self.model.addMVar(m, lb = -1*GRB.INFINITY)
        self.model.addConstr(w_diff == self.w - w_prev)
        return (1 / 2) * (w_diff @ w_diff)

    def print_var_info(self, names=None):
        # dictionary of variables and thier names
        if names is None:
            names = {}
        for key in names.keys():
            print(key, names[key])
        print("x", self.w.X)
        print("z", self.b.X)
        print("xi", self.xi.X)
        print("")

    def set_model(self, svm_constrs=None, delta=0, w_prev_soln=None):
        # remove constraints
        if svm_constrs is None:
            svm_constrs = []
        self.model.remove(self.model.getConstrs())
        # parameter definitions
        n, m = self.exogenous.shape

        epsilon = self.epsilon
        # objective function components

        if svm_constrs:
            for con in svm_constrs:
                self.model.addConstr(con, 'target')

        if w_prev_soln is not None:
            hyperplane_penalty = (1-delta)*self.svm_margin + delta*self.svm_change(w_prev_soln)
        else:
            hyperplane_penalty = self.svm_margin

        if type(self.soft_margin) is np.ndarray and np.max(self.soft_margin) > 10 ** 6:  # not the first solve
            big_penalty = np.max(self.soft_margin)
            normalized_margin = self.soft_margin / big_penalty
            print("big penalty mode")
            self.model.setObjective((1 / big_penalty) * hyperplane_penalty +
                                    (1 / n) * normalized_margin[:, 0] @ self.xi,
                                    GRB.MINIMIZE)
        elif type(self.soft_margin) is not np.ndarray and self.soft_margin > 10 ** 6:
            big_penalty = self.soft_margin
            print("big penalty mode")
            self.model.setObjective((1 / big_penalty) * hyperplane_penalty +
                                    (1 / n) * (self.xi.sum()), GRB.MINIMIZE)
        else:
            self.model.setObjective(hyperplane_penalty +
                                    self.soft_penalty, GRB.MINIMIZE)

        for i in range(n):
            y_i = self.exogenous.iloc[i].values
            self.model.addConstr((2 * self.mvo_z[i] - 1) * (y_i @ self.w + self.b) + self.xi[i] >= self.epsilon)
            # abs_b_ = self.abs_b.tolist()
            # b_ = self.b.tolist()
            # self.model.addGenConstrAbs(abs_b_[0], b_[0], "absconstr")
            # self.model.addConstr(abs_b_[0] <= 0.1*self.epsilon)
            # self.model.addConstr(self.w @ y_i + self.b <= (-1)*self.epsilon
            # + self.xi[i] + big_m*self.mvo_z[i], "svm1" )

    def optimize(self, cbb=None):

        if cbb not in [None]:
            self.model._cur_obj = float('inf')
            self.model._time = time.time()
            self.model.optimize(callback=cbb)
        else:
            self.model.optimize()
        self.model.write('portfolio_selection_optimization.lp')


def check_partial_min(instance, w_prev):
    """checks for partial min"""
    allg0 = np.all(w_prev > 10 ** (-9))
    relative_diff = np.all(np.abs((instance.SVM_.w.x - w_prev) / w_prev) < 0.05)
    matching_penalty = np.abs(instance.MVO_.xi.x - instance.SVM_.xi.x).sum() <= 10 ** (-6)
    return (np.abs(instance.SVM_.w.x - w_prev).sum() < 10 ** (-12)) or (allg0 and relative_diff) and matching_penalty \
           or (instance.SVM_.xi.x.sum() + instance.MVO_.xi.x.sum() < 10 ** (-9))


def check_global_convergence(instance, w_param_init, z_param_init = None, x_param_init = None, change_threshold = 0.2):
    """checks for global convergence"""
    if z_param_init is not None:
        big_x_old = (x_param_init > 1e-4).astype(int)
        big_x_new = (instance.MVO_.x.x > 1e-4).astype(int)
        z_changed = np.abs(big_x_old - big_x_new).sum()/len(z_param_init) > change_threshold
    else:
        z_changed = False
    allg0 = np.all(w_param_init > 10 ** (-12))
    relative_diff = np.all(np.abs((instance.SVM_.w.x - w_param_init) / w_param_init) < 0.05)
    return (np.abs(instance.SVM_.w.x - w_param_init).sum() < 10 ** (-12) or (allg0 and relative_diff)) \
           and instance.SVM_.xi.x.sum() + instance.MVO_.xi.x.sum() < 10 ** (-9) or z_changed


def get_multiplier(instance):
    """
    gets the penalty update for the instance
    :param instance:
    :return:
    """
    svm_multiplier = (instance.SVM_.xi.x > 10 ** (-12)).astype(int).reshape(instance.MVO_.mean_ret.shape)
    mvo_multiplier = (instance.MVO_.xi.x > 10 ** (-12)).astype(int).reshape(instance.MVO_.mean_ret.shape)
    all_multiplier = (instance.SVM_.xi.x.sum() + instance.MVO_.xi.x.sum() > 10 ** (-9)).astype(int)
    if type(instance.SVM_.soft_margin) is np.ndarray:
        mult = (1 + 0.5 * all_multiplier) * np.ones_like(
            instance.SVM_.soft_margin) + 0.25 * svm_multiplier + 0.25 * mvo_multiplier
    else:
        mult = 2
    return mult


class SVM_MVO_ADM:
    # '''this class models the integrated SVM MVO problem using the ADM solution method'''
    big_m = 100
    epsilon = 0.001

    def __init__(self, MVO_, SVM_, IterLim=20, ParamLim=10):
        self.MVO_ = MVO_
        self.SVM_ = SVM_
        self.IterLim = IterLim
        self.ParamLim = ParamLim
        self.x = None
        self.z = None
        self.w = None
        self.b = None
        self.xi_svm = None
        self.xi_mvo = None

    @property
    def describe(self):
        desc = "SVM MVO with Alternating Direction Method"
        shrt = "SVM MVO ADM"
        return desc, shrt

    @property
    def portfolio_risk(self):
        return self.MVO_.x @ self.MVO_.cov @ self.MVO_.x

    @property
    def port_exptd_ret(self):
        return self.MVO_.mean_ret[:, 0] @ self.MVO_.x

    @property
    def soft_penalty(self):
        n, m = self.SVM_.exogenous.shape
        return self.SVM_.soft_penalty

    @property
    def soft_penalty_mvo(self):

        return self.MVO_.soft_penalty

    @property
    def svm_margin(self):
        return (1 / 2) * (self.SVM_.w @ self.SVM_.w)

    @property
    def tics(self):
        return self.MVO_.tics

    @property
    def objective_svm(self):
        return self.portfolio_risk.getValue() + self.svm_margin.getValue() + self.soft_penalty.getValue()

    @property
    def objective_mvo(self):
        return self.portfolio_risk.getValue() + self.svm_margin.getValue() + self.soft_penalty_mvo.getValue()

    def initialize_soln(self, set_return=True, constrs=None, svm_constrs=None, warm_starts=None, delta=0, w_prev_soln=None):
        if svm_constrs is None:
            svm_constrs = []
        if constrs is None:
            constrs = []
        self.MVO_.soft_margin = 0  # on initiliation make sure that the oftmargin is 0
        self.MVO_.set_model(set_return, constrs, warm_starts)  # set up the model
        self.MVO_.optimize()  # find optimal solution
        if self.MVO_.model.status == 4:
            return  # return threshold must be reduced
        self.SVM_.mvo_z = self.MVO_.z.x
        self.SVM_.set_model(svm_constrs, delta, w_prev_soln)
        self.SVM_.optimize()

    def solve_adm(self, store_data=True, set_return=True, constrs=None, svm_constrs=None, delta=0, w_prev_soln=None):
        if svm_constrs is None:
            svm_constrs = []
        if constrs is None:
            constrs = []
        ws = []
        xs = []
        zs = []
        penalty_hist = []
        c = self.SVM_.soft_margin / (2 ** (self.ParamLim))  # initialized to a number > 0
        self.SVM_.soft_margin, self.MVO_.soft_margin = (c, c)
        xi_mvo = []
        xi_svm = []
        objectives_svm, objectives_mvo = ([], [])
        start = time.time()
        end = time.time()
        z_param_init = self.MVO_.z.x
        x_param_init = self.MVO_.x.x
        for k in range(self.ParamLim):

            i, converged = (0, False)

            w_param_init = self.SVM_.w.x

            while (i <= self.IterLim) and (not converged):
                w_prev = self.SVM_.w.x
                x_prev = self.MVO_.x.x
                z_prev = self.MVO_.z.x
                if store_data:
                    ws.append(self.SVM_.w.x)
                    xs.append(self.MVO_.x.x)
                    zs.append(self.MVO_.z.x)
                    xi_mvo.append(self.MVO_.xi.x)
                    xi_svm.append(self.SVM_.xi.x)
                    objectives_svm.append(self.objective_svm[0])
                    objectives_mvo.append(self.objective_mvo[0])
                    penalty_hist.append(self.SVM_.soft_margin)
                self.MVO_.svm_b = self.SVM_.b.x
                self.MVO_.svm_w = self.SVM_.w.x

                self.MVO_.set_model(set_return, constrs, warm_starts=[x_prev, z_prev])
                self.MVO_.optimize()

                self.SVM_.mvo_z = self.MVO_.z.x
                self.SVM_.set_model(svm_constrs, delta, w_prev_soln)
                self.SVM_.optimize()
                i += 1

                if check_partial_min(self, w_prev):
                    converged = True
                    #print("partial min convergence")
                    if store_data:
                        ws.append(self.SVM_.w.x)
                        xs.append(self.MVO_.x.x)
                        zs.append(self.MVO_.z.x)
                        xi_mvo.append(self.MVO_.xi.x)
                        xi_svm.append(self.SVM_.xi.x)
                        objectives_svm.append(self.objective_svm[0])
                        objectives_mvo.append(self.objective_mvo[0])
                        penalty_hist.append(self.SVM_.soft_margin)
                end = time.time()

            if check_global_convergence(self, w_param_init, z_param_init, x_param_init):
                print("ADM terminated with C = ", np.mean(self.SVM_.soft_margin))
                break

            mult = get_multiplier(self)
            self.SVM_.soft_margin, self.MVO_.soft_margin = (self.SVM_.soft_margin * mult, self.MVO_.soft_margin * mult)
            #print("outer iteration ", k)
            #print(np.max(self.SVM_.soft_margin))

        self.x = self.MVO_.x
        self.z = self.MVO_.z
        self.w = self.SVM_.w
        self.b = self.SVM_.b
        self.xi_svm = self.SVM_.xi
        self.xi_mvo = self.MVO_.xi
        self.SVM_.soft_margin, self.MVO_.soft_margin = (
            c * (2 ** (self.ParamLim)), c * (2 ** (self.ParamLim)))  # reinitialize C
        return np.array(ws), np.array(xs), np.array(zs), np.array(xi_mvo), np.array(
            xi_svm), end - start, objectives_svm, objectives_mvo, np.array(penalty_hist)

    def evaluate(self, realized_ret):
        ret = np.dot(self.x.X, realized_ret)
        return ret  # use this to calculate out of sample rets and var

    def get_estimates(self):
        vol_metric = np.sqrt(self.MVO_.portfolio_risk.getValue())[0]
        expt_ret_metric = self.MVO_.port_exptd_ret.getValue()[0]
        return [vol_metric, expt_ret_metric]  # use this for efficient frontiers

    def silence_output(self):
        self.MVO_.model.Params.LogtoConsole = 0
        self.SVM_.model.Params.LogtoConsole = 0

    def get_results(self, export_dir='', fig_size=()):

        lng, shrt = self.describe
        vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
        expt_ret_metric = self.port_exptd_ret.getValue()[0]
        results = pd.DataFrame(data=np.append(self.x.X, [vol_metric, expt_ret_metric]),
                               index=list(self.tics) + ['Volatility', 'Expected Return'], columns=[lng])

        if export_dir != '':
            results.to_csv(export_dir + 'results.csv')

        if fig_size != () and type(fig_size) in [list, tuple]:
            results[:-2].plot.bar(figsize=fig_size)

        return results.transpose()

    def get_frontier(self, export_dir='', fig_size=(10, 8), lower_ret=None, upper_ret=None):

        n, m = self.SVM_.exogenous.shape
        f = 25
        mean_ret = self.MVO_.mean_ret[:, 0]
        # F is the number of portfolios to use for frontier
        frontier = np.empty((2, f))
        # ws will contain the w's and b values for each portfolio 
        ws = np.empty((f, m + 1))
        # xis will contain the w's and b values for each portfolio 
        xis = np.empty((f, n))
        # targets for returns
        if lower_ret is None:
            lower_ret = mean_ret.min()
        if upper_ret is None:
            upper_ret = mean_ret.max()
        ret_targ = np.linspace(lower_ret, upper_ret, f)

        for i in range(f):

            constraints = [self.MVO_.port_exptd_ret == ret_targ[i]]
            self.initialize_soln(set_return=False, constrs=constraints)
            self.solve_adm(store_data=False, set_return=False, constrs=constraints)

            if self.MVO_.model.status == 4:
                break

            vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
            frontier[:, i] = np.array([vol_metric, ret_targ[i]])
            ws[i, :] = np.concatenate([self.w.x, self.b.x])
            xis[i, :] = self.MVO_.xi.x

        if self.MVO_.model.status == 4:
            print("Resolving Model to initial state (return target) then exiting")
            self.MVO_.model.remove(self.MVO_.ret_target)
            self.MVO_.model.update()
            self.MVO_.ret_target = self.MVO_.model.addConstr(self.MVO_.port_exptd_ret >= self.MVO_.ret_constr, 'target')
            self.MVO_.model.optimize()
            return (None, None, None)

        # restore model to original state
        self.MVO_.model.remove(self.MVO_.ret_target)
        self.MVO_.model.update()
        self.MVO_.ret_target = self.MVO_.model.addConstr(self.MVO_.port_exptd_ret >= self.MVO_.ret_constr, 'target')
        self.initialize_soln()
        self.solve_adm()

        fig, ax = plt.subplots(figsize=fig_size)
        # Plot efficient frontier
        ax.plot(frontier[0], frontier[1], '-*', label='Efficient Frontier', color='DarkGreen')

        # Format and display the final plot
        ax.axis([frontier[0].min() * 0.7, frontier[0].max() * 1.3, mean_ret.min() * 1.2, mean_ret.max() * 1.2])
        ax.set_xlabel('Volatility (standard deviation)')
        ax.set_ylabel('Expected Return')
        # ax.legend()
        ax.grid()
        plt.show()
        if (export_dir != ''):
            plt.savefig(export_dir + "EfficientFrontier.png")
        return (frontier, ws, xis)


# v2 only updates the penalties for the assets that are not SVM constrained
class SVM_MVO_ADM_v2:
    # '''this class models the integrated SVM MVO problem using the ADM solution method'''
    big_m = 100
    epsilon = 0.001

    def __init__(self, MVO_, SVM_, IterLim=200, ParamLim=10):
        self.MVO_ = MVO_
        self.SVM_ = SVM_
        self.IterLim = IterLim
        self.ParamLim = ParamLim
        self.x = None
        self.z = None
        self.w = None
        self.b = None
        self.xi_svm = None
        self.xi_mvo = None

    @property
    def describe(self):
        desc = "SVM MVO with Alternating Direction Method"
        shrt = "SVM MVO ADM"
        return desc, shrt

    @property
    def portfolio_risk(self):
        return self.MVO_.x @ self.MVO_.cov @ self.MVO_.x

    @property
    def port_exptd_ret(self):
        return self.MVO_.mean_ret[:, 0] @ self.MVO_.x

    @property
    def soft_penalty(self):

        return self.SVM_.soft_penalty

    @property
    def soft_penalty_mvo(self):

        return self.MVO_.soft_penalty

    @property
    def svm_margin(self):
        return (1 / 2) * (self.SVM_.w @ self.SVM_.w)

    @property
    def tics(self):
        return self.MVO_.tics

    @property
    def objective_svm(self):
        return self.portfolio_risk.getValue() + self.svm_margin.getValue() + self.soft_penalty.getValue()

    @property
    def objective_mvo(self):
        return self.portfolio_risk.getValue() + self.svm_margin.getValue() + self.soft_penalty_mvo.getValue()

    def initialize_soln(self, set_return=True, constrs=None, svm_constrs=None, zero_soft=True):
        if svm_constrs is None:
            svm_constrs = []
        if constrs is None:
            constrs = []
        if zero_soft:
            self.MVO_.soft_margin = 0  # on initiliation make sure that the oftmargin is 0
        self.MVO_.set_model(set_return, constrs)  # set up the model
        self.MVO_.optimize()  # find optimal solution
        self.SVM_.mvo_z = self.MVO_.z.x
        self.SVM_.set_model(svm_constrs)
        self.SVM_.optimize()

    def solve_adm(self, store_data=True, set_return=True, constrs=None, svm_constrs=None):
        if svm_constrs is None:
            svm_constrs = []
        if constrs is None:
            constrs = []
        ws = []
        xs = []
        zs = []

        xi_mvo = []
        xi_svm = []
        objectives_svm, objectives_mvo = ([], [])
        start = time.time()

        c = self.SVM_.soft_margin / (2 ** (self.ParamLim))  # initialized to a number > 0
        old_cov = self.MVO_.cov
        self.SVM_.soft_margin, self.MVO_.soft_margin = (c, c)
        stalled_previously = False
        xi_previous = 10 ** 6

        for k in range(self.ParamLim):
            i, converged = (0, False)

            w_param_init = self.SVM_.w.x
            x_param_init = self.MVO_.x.x

            while (i <= self.IterLim) and (not converged):
                w_prev = self.SVM_.w.x
                if store_data:
                    ws.append(self.SVM_.w.x)
                    xs.append(self.MVO_.x.x)
                    zs.append(self.MVO_.z.x)
                    xi_mvo.append(self.MVO_.xi.x)
                    xi_svm.append(self.SVM_.xi.x)
                    objectives_svm.append(self.objective_svm[0])
                    objectives_mvo.append(self.objective_mvo[0])

                self.MVO_.svm_b = self.SVM_.b.x
                self.MVO_.svm_w = self.SVM_.w.x

                self.MVO_.set_model(set_return, constrs)
                self.MVO_.optimize()

                self.SVM_.mvo_z = self.MVO_.z.x
                self.SVM_.set_model(svm_constrs)
                self.SVM_.optimize()
                i += 1

                if check_partial_min(self, w_prev):
                    converged = True
                    if store_data:
                        ws.append(self.SVM_.w.x)
                        xs.append(self.MVO_.x.x)
                        zs.append(self.MVO_.z.x)
                        xi_mvo.append(self.MVO_.xi.x)
                        xi_svm.append(self.SVM_.xi.x)
                        objectives_svm.append(self.objective_svm[0])
                        objectives_mvo.append(self.objective_mvo[0])

            # out of inner loop
            x_partial = self.MVO_.x.x

            if np.abs(x_partial - x_param_init).sum() <= 10 ** (-12) \
                    and (self.SVM_.xi.x.sum() + self.MVO_.xi.x.sum() >= 10 ** (-6)):  # infeasible
                print("Stalling at Infeasible Point")
                # if it is not the first stall and the solution has become more feasible 
                # update our best starts

                if not stalled_previously:
                    best_b = old_cov  # the best start is this B
                    # Best_SVM_Margin, Best_MVO_Margin = self.SVM_.soft_margin , self.MVO_.soft_margin
                if stalled_previously and self.SVM_.xi.x.sum() < xi_previous:
                    best_b = B  # the best start is this B
                    # Best_SVM_Margin, Best_MVO_Margin = self.SVM_.soft_margin , self.MVO_.soft_margin
                print('SVM ', self.SVM_.xi.x.sum())
                print('MVO ', self.MVO_.xi.x.sum())
                # reset penalty
                # self.SVM_.soft_margin , self.MVO_.soft_margin =  (C, C)
                # generate random starts 
                u = np.random.rand()
                A = np.random.rand(len(self.MVO_.x.x), len(self.MVO_.x.x))
                B = u * np.dot(A, A.transpose()) + (1 - u) * np.identity(len(self.MVO_.x.x))

                if k == self.ParamLim - 1:  # last iteration
                    print('using the best start')
                    self.MVO_.cov = best_b
                    # self.SVM_.soft_margin , self.MVO_.soft_margin = Best_SVM_Margin, Best_MVO_Margin
                else:
                    self.MVO_.cov = B
                    xi_previous = self.SVM_.xi.x.sum() + self.MVO_.xi.x.sum()

                self.initialize_soln(zero_soft=False)
                # now optimize the true covariance from this point
                self.MVO_.cov = old_cov
            # out of inner loop 
            w_update = self.SVM_.w.x

            if check_global_convergence(self, w_param_init):
                print("ADM terminated with C = ", self.SVM_.soft_margin)
                break
                # if we have not terminated the update the margins based on the ones with xi > 0

            mult = get_multiplier(self)
            self.SVM_.soft_margin, self.MVO_.soft_margin = (self.SVM_.soft_margin * mult, self.MVO_.soft_margin * mult)

        end = time.time()
        self.x = self.MVO_.x
        self.z = self.MVO_.z
        self.w = self.SVM_.w
        self.b = self.SVM_.b
        self.xi_svm = self.SVM_.xi
        self.xi_mvo = self.MVO_.xi
        self.SVM_.soft_margin, self.MVO_.soft_margin = (c * (2 ** self.ParamLim), c * (2 ** self.ParamLim))

        return np.array(ws), np.array(xs), np.array(zs), np.array(xi_mvo), np.array(
            xi_svm), end - start, objectives_svm, objectives_mvo

    def evaluate(self, realized_ret):
        ret = np.dot(self.x.X, realized_ret)
        return ret  # use this to calculate out of sample rets and var

    def get_estimates(self):
        vol_metric = np.sqrt(self.MVO_.portfolio_risk.getValue())[0]
        expt_ret_metric = self.MVO_.port_exptd_ret.getValue()[0]
        return [vol_metric, expt_ret_metric]  # use this for efficient frontiers

    def silence_output(self):
        self.MVO_.model.Params.LogtoConsole = 0
        self.SVM_.model.Params.LogtoConsole = 0

    def get_results(self, export_dir='', fig_size=()):

        lng, shrt = self.describe
        vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
        expt_ret_metric = self.port_exptd_ret.getValue()[0]
        results = pd.DataFrame(data=np.append(self.x.X, [vol_metric, expt_ret_metric]),
                               index=list(self.tics) + ['Volatility', 'Expected Return'], columns=[lng])

        if export_dir != '':
            results.to_csv(export_dir + 'results.csv')

        if fig_size != () and type(fig_size) in [list, tuple]:
            results[:-2].plot.bar(figsize=fig_size)

        return results.transpose()

    def get_frontier(self, export_dir='', fig_size=(10, 8), lower_ret=None, upper_ret=None):

        n, m = self.SVM_.exogenous.shape
        f = 25
        mean_ret = self.MVO_.mean_ret[:, 0]
        # F is the number of portfolios to use for frontier
        frontier = np.empty((2, f))
        # ws will contain the w's and b values for each portfolio 
        ws = np.empty((f, m + 1))
        # xis will contain the w's and b values for each portfolio 
        xis = np.empty((f, n))
        # targets for returns
        if lower_ret is None:
            lower_ret = mean_ret.min()
        if upper_ret is None:
            upper_ret = mean_ret.max()
        ret_targ = np.linspace(lower_ret, upper_ret, f)

        for i in range(f):

            constraints = [self.MVO_.port_exptd_ret == ret_targ[i]]
            self.initialize_soln(set_return=False, constrs=constraints)
            self.solve_adm(store_data=False, set_return=False, constrs=constraints)

            if self.MVO_.model.status == 4:
                break

            vol_metric = np.sqrt(self.portfolio_risk.getValue())[0]
            frontier[:, i] = np.array([vol_metric, ret_targ[i]])
            ws[i, :] = np.concatenate([self.w.x, self.b.x])
            xis[i, :] = self.MVO_.xi.x

        if self.MVO_.model.status == 4:
            print("Resolving Model to initial state (return target) then exiting")
            self.MVO_.model.remove(self.MVO_.ret_target)
            self.MVO_.model.update()
            self.MVO_.ret_target = self.MVO_.model.addConstr(self.MVO_.port_exptd_ret >= self.MVO_.ret_constr, 'target')
            self.MVO_.model.optimize()
            return None, None, None

        # restore model to original state
        self.MVO_.model.remove(self.MVO_.ret_target)
        self.MVO_.model.update()
        self.MVO_.ret_target = self.MVO_.model.addConstr(self.MVO_.port_exptd_ret >= self.MVO_.ret_constr, 'target')
        self.initialize_soln()
        self.solve_adm()

        fig, ax = plt.subplots(figsize=fig_size)
        # Plot efficient frontier
        ax.plot(frontier[0], frontier[1], '-*', label='Efficient Frontier', color='DarkGreen')

        # Format and display the final plot
        ax.axis([frontier[0].min() * 0.7, frontier[0].max() * 1.3, mean_ret.min() * 1.2, mean_ret.max() * 1.2])
        ax.set_xlabel('Volatility (standard deviation)')
        ax.set_ylabel('Expected Return')
        # ax.legend()
        ax.grid()
        plt.show()
        if export_dir != '':
            plt.savefig(export_dir + "EfficientFrontier.png")
        return frontier, ws, xis
