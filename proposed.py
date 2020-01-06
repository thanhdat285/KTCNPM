#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas 
import glob
import sys 
import matplotlib.pyplot as plt
from utils import find_column_indices
beta_static_lookup = {
    "C2011-05": 0.455,
    "C2011-07": 0.455,
    "C2011-12": 0.187,
    "C2011-13": 0.012,
    "C2012-13": 0.108,
    "C2013-01": 0.232,
    "C2013-02": 0.140,
    "C2013-03": 0.359,
    "C2013-04": 1.0,
    "C2013-06": 0.817,
    "C2013-07": 0.0,
    "C2013-08": 1.0,
    "C2013-09": 0.92,
    "C2013-10": 1.0,
    "C2013-11": 0.014,
    "C2013-12": 0.32,
    "C2013-13": 0.000,
    "C2013-15": 0.099,
    "C2014-04": 0.223,
    "C2014-05": 0.071,
    "C2014-06": 0.025,
    "C2014-07": 0.444,
    "C2014-08": 0.975
}


data_file = sys.argv[1]
dfs = pandas.read_excel(f'data/{data_file}.xlsx', sheet_name=None,
    skiprows=1)
sheetnames = list(dfs.keys())



def parse_date(date):
    elms = date.split()
    total_hour = 0
    for elm in elms:
        if elm.endswith("d"):
            total_hour += int(elm[:-1])*24
        elif elm.endswith("h"):
            total_hour += int(elm[:-1])
    return total_hour

import numpy as np
def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue) 
    ypred = np.array(ypred)
    error = np.abs((ytrue-ypred)/ytrue) * 100
    mean_error = np.mean(error)
    return mean_error, error


# In[9]:


import os 
if not os.path.isdir("data"):
    os.makedirs("data")
baselines = dfs['Baseline Schedule'][['ID', 'Duration', 'Total Cost']].values
baselines[:,1] = [parse_date(x) for x in baselines[:,1]]
# planned duration
BAC = baselines[0,2]
tracking_periods = [x for x in sheetnames if "TP" in x]
n_tracking_periods = baselines[0,1] / (20*60)
print("BAC:", BAC)
print("Number of tracking periods:", n_tracking_periods)


def cost_forecasting_evm():
    # Col 0 = ID, col 12 = Duration
    beta = beta_static_lookup[data_file]
    ACs = [0] # init AT0 = 0
    EVs = [0]
    EAC_costs = [] # predict project duration
    start_test = False
    for period in tracking_periods:
        print("Tracking periods:", period)
        cols = find_column_indices(dfs[period].values[1], ["ID", "Actual Cost", "Earned Value (EV)", "Planned Value (PV)"])
        data_period = dfs[period].values[2:, cols] 
        assert (baselines[:,0] == data_period[:,0]).sum() == len(baselines), "Wrong permutation!"

        AC = data_period[0, 1]
        ACs.append(AC)
        EV = data_period[0, 2]
        PV = data_period[0, 3]
        # if EV < PV and not start_test:
        #     start_test = True
        # if start_test:
        if True:
            CPI = EV/AC
            EAC = (BAC-EV) / CPI + AC
            EAC_costs.append(EAC)
    print("Project actual costs: ", data_period[0,1])
    mape, error = MAPE([ACs[-1]]*len(EAC_costs[:-1]), EAC_costs[:-1])
    print("EVM MAPE: ", mape)
    # plt.plot(error)
    # plt.savefig(f"{data_file}-static.png")
    return error, mape


def cost_forecasting():
    # init trend
    Ts_AC = [BAC/n_tracking_periods]
    Ts_EV = [BAC/n_tracking_periods]
    print("T0_AC = T0_EV: ", Ts_AC[0])

    # Col 0 = ID, col 12 = Duration
    beta = beta_static_lookup[data_file]
    ACs = [0] # init AT0 = 0
    t = 1
    EVs = [0]
    EAC_costs = [] # predict project duration
    start_test = False
    for period in tracking_periods:
        print("Tracking periods:", period)
        cols = find_column_indices(dfs[period].values[1], ["ID", "Actual Cost", "Earned Value (EV)", "Planned Value (PV)"])
        data_period = dfs[period].values[2:, cols] 
        assert (baselines[:,0] == data_period[:,0]).sum() == len(baselines), "Wrong permutation!"

        # current trend
        cur_AC = data_period[0,1]
        ACs.append(cur_AC)
        T_AC = beta*(ACs[t] - ACs[t-1]) + (1-beta)*Ts_AC[t-1]
        Ts_AC.append(T_AC)

        EV = data_period[0,2]
        PV = data_period[0,3]
        EVs.append(EV)
        T_EV = beta*(EVs[t] - EVs[t-1]) + (1-beta)*Ts_EV[t-1]
        Ts_EV.append(T_EV)

        # if EV < PV and not start_test:
        #     start_test = True
        # if start_test:
        # if t >= (len(tracking_periods)*1/2) and T_EV > 0:
        # if T_EV > 0:
        # if t >= (len(tracking_periods)*2/3) and T_EV > 0:
        if T_EV > 0:
            k = (BAC-EVs[t]) / T_EV
            EAC = ACs[t] + k * T_AC
            EAC_costs.append(EAC)
            print("EAC:", EAC, k, T_AC)
        # end calculate
        t += 1
    print("Project actual costs: ", data_period[0,1])
    mape, error = MAPE([ACs[-1]]*len(EAC_costs[:-1]), EAC_costs[:-1])
    print("MAPE: ", mape)
    # plt.plot(error)
    # plt.savefig(f"{data_file}-static.png")
    return error, mape

# # Dynamic Beta

# In[10]:

def dynamic_cost():
    # init trend
    Ts_AC = [BAC/n_tracking_periods]
    Ts_EV = [BAC/n_tracking_periods]
    init_T = BAC/n_tracking_periods
    print("T0_AC = T0_EV: ", Ts_AC[0])

    def select_best_beta(cur_AC):
        betas = [] # list of tuples (beta, MAPE)
        for beta in np.arange(0.0, 1, 0.05): # e^beta*x
            _ACs = [0]
            _Ts_AC = [0]
            predict_ACs = []
            for prev_period in range(0, cur_period):
                # predict AC of current period, cur_AC = prev_AC + trend_AC
                data_prev_period = dfs[tracking_periods[prev_period]].values[2:, cols]
                prev_AC = data_prev_period[0, 1]
                _ACs.append(prev_AC)
                # _T_AC = beta*(_ACs[prev_period] - _ACs[prev_period-1]) + (1-beta) * _Ts_AC[prev_period-1]
                _T_AC = calculate_current_trend(_ACs, beta, init_T)
                # _Ts_AC.append(_T_AC)
                # predict_AC = _ACs[prev_period-1] + (cur_period - prev_period)*_Ts_AC[prev_period-1]
                predict_AC = _ACs[prev_period-1] + (cur_period - prev_period)*_T_AC
                predict_ACs.append(predict_AC)
            if len(predict_ACs) == 0:
                error = 0
            else:
                ytrue = np.array([cur_AC]*len(predict_ACs))
                ypred = np.array(predict_ACs)
                error = np.abs((ytrue-ypred)/ytrue) * 100
                weights = 1 - np.arange(0, 1, 1/len(error))[:len(error)]
                # weights = np.zeros(len(error))
                # weights[-1] = 1
                error = np.sum(error*weights)
            # error = np.mean(error)

            # error = MAPE([cur_AC]*len(predict_ACs), [predict_AC])
            # prev_period = cur_period - 1
            # # _T_AC = beta*(_ACs[prev_period] - _ACs[prev_period-1]) + (1-beta) * _Ts_AC[prev_period-1]
            # _T_AC = calculate_current_trend(_ACs, beta)
            # predict_AC = _ACs[-1] + _T_AC
            # error = MAPE([cur_AC], [predict_AC])
            betas.append((beta, error))
        # select best beta
        beta = sorted(betas, key=lambda x: x[1])[0][0]
        return beta

    def calculate_current_trend(ACs, beta, init_T):
        # ACs[0] must be equals to 0
        if len(ACs) == 1: return init_T
        prev_T = calculate_current_trend(ACs[:-1], beta, init_T)
        T = beta*(ACs[-1] - ACs[-2]) + (1-beta) * prev_T
        return T

    # Col 0 = ID, col 12 = Duration
    ACs = [0] # init AT0 = 0
    t = 1
    EVs = [0]
    EAC_costs = [] # predict project duration
    start_test = False
    cols = find_column_indices(dfs[tracking_periods[0]].values[1], ["ID", "Actual Cost", "Earned Value (EV)", "Planned Value (PV)"])
    betas = []
    for cur_period, period in enumerate(tracking_periods):
        print("=== Tracking periods:", period)
        data_period = dfs[period].values[2:, cols]
        cur_AC = data_period[0, 1]
        ACs.append(cur_AC)
        # find optimal beta
        beta = select_best_beta(cur_AC)
        print("Best beta", beta)
        # current trend
        # T_AC = beta*(ACs[t] - ACs[t-1]) + (1-beta)*Ts_AC[t-1]
        T_AC = calculate_current_trend(ACs, beta, init_T)
        Ts_AC.append(T_AC)

        EV = data_period[0,2]
        PV = data_period[0,3]
        EVs.append(EV)
        T_EV = beta*(EVs[t] - EVs[t-1]) + (1-beta)*Ts_EV[t-1]
        Ts_EV.append(T_EV)

        # if EV < PV and not start_test:
        #     start_test = True
        # if start_test:
        # if t >= (len(tracking_periods)*3/4) and T_EV > 0:
        if T_EV > 0:
            betas.append(beta)
            k = (BAC-EVs[t]) / T_EV
            EAC = ACs[t] + k * T_AC
            EAC_costs.append(EAC)
            print(f"EAC: {EAC:.3f}\t{k}\t{T_AC}")
        # end calculate
        t += 1
    print("Project actual costs: ", ACs[-1])
    mape, error = MAPE([ACs[-1]]*len(EAC_costs[:-1]), EAC_costs[:-1])
    print(f"Dynamic MAPE: {mape:.2f}")
    # plt.figure()
    # plt.plot(betas)
    # plt.xlabel("tracking periods")
    # plt.xticks(np.arange(len(betas)))
    # plt.ylabel("beta")
    # plt.savefig(f"figures/{data_file}-beta.png")
    return error, mape, ACs[-len(error):], EAC_costs[:-1], betas[:-1]

def dynamic_cost_without_recursive():
    # init trend
    Ts_AC = [BAC/n_tracking_periods]
    Ts_EV = [BAC/n_tracking_periods]
    init_T = BAC/n_tracking_periods
    print("T0_AC = T0_EV: ", Ts_AC[0])

    def select_best_beta(cur_AC):
        betas = [] # list of tuples (beta, MAPE)
        for beta in np.arange(0.0, 1, 0.05): # e^beta*x
            _ACs = [0]
            _Ts_AC = [0]
            predict_ACs = []
            for prev_period in range(0, cur_period):
                # predict AC of current period, cur_AC = prev_AC + trend_AC
                data_prev_period = dfs[tracking_periods[prev_period]].values[2:, cols]
                prev_AC = data_prev_period[0, 1]
                _ACs.append(prev_AC)
                _T_AC = beta*(_ACs[prev_period] - _ACs[prev_period-1]) + (1-beta) * _Ts_AC[prev_period-1]
                # _T_AC = calculate_current_trend(_ACs, beta, init_T)
                # _Ts_AC.append(_T_AC)
                # predict_AC = _ACs[prev_period-1] + (cur_period - prev_period)*_Ts_AC[prev_period-1]
                predict_AC = _ACs[prev_period-1] + (cur_period - prev_period)*_T_AC
                predict_ACs.append(predict_AC)
            if len(predict_ACs) == 0:
                error = 0
            else:
                ytrue = np.array([cur_AC]*len(predict_ACs))
                ypred = np.array(predict_ACs)
                error = np.abs((ytrue-ypred)/ytrue) * 100
                # weights = np.abs(0.5 - np.arange(0, 1, 1/len(error)))
                weights = 1 - np.arange(0, 1, 1/len(error))[:len(error)]
                # weights = 1/(1+np.exp(-np.arange(0,1,1/len(error)))) - 0.5
                # weights = np.ones(len(error))
                # error = np.sum(error*weights) / weights.sum()
                error = np.sum(error*weights)
            # error = np.mean(error)

            # error = MAPE([cur_AC]*len(predict_ACs), [predict_AC])
            # prev_period = cur_period - 1
            # # _T_AC = beta*(_ACs[prev_period] - _ACs[prev_period-1]) + (1-beta) * _Ts_AC[prev_period-1]
            # _T_AC = calculate_current_trend(_ACs, beta)
            # predict_AC = _ACs[-1] + _T_AC
            # error = MAPE([cur_AC], [predict_AC])
            betas.append((beta, error))
        # select best beta
        beta = sorted(betas, key=lambda x: x[1])[0][0]
        return beta

    def calculate_current_trend(ACs, beta, init_T):
        # ACs[0] must be equals to 0
        if len(ACs) == 1: return init_T
        prev_T = calculate_current_trend(ACs[:-1], beta, init_T)
        T = beta*(ACs[-1] - ACs[-2]) + (1-beta) * prev_T
        return T

    # Col 0 = ID, col 12 = Duration
    ACs = [0] # init AT0 = 0
    t = 1
    EVs = [0]
    EAC_costs = [] # predict project duration
    start_test = False
    cols = find_column_indices(dfs[tracking_periods[0]].values[1], ["ID", "Actual Cost", "Earned Value (EV)", "Planned Value (PV)"])
    betas = []
    for cur_period, period in enumerate(tracking_periods):
        print("=== Tracking periods:", period)
        data_period = dfs[period].values[2:, cols]
        cur_AC = data_period[0, 1]
        ACs.append(cur_AC)
        # find optimal beta
        beta = select_best_beta(cur_AC)
        print("Best beta", beta)
        # current trend
        # T_AC = beta*(ACs[t] - ACs[t-1]) + (1-beta)*Ts_AC[t-1]
        T_AC = calculate_current_trend(ACs, beta, init_T)
        Ts_AC.append(T_AC)

        EV = data_period[0,2]
        PV = data_period[0,3]
        EVs.append(EV)
        T_EV = beta*(EVs[t] - EVs[t-1]) + (1-beta)*Ts_EV[t-1]
        Ts_EV.append(T_EV)

        if EV < PV and not start_test:
            start_test = True
        if start_test:
            betas.append(beta)
            k = (BAC-EVs[t]) / T_EV
            EAC = ACs[t] + k * T_AC
            EAC_costs.append(EAC)
            print(f"EAC: {EAC:.3f}\t{k}\t{T_AC}")
        # end calculate
        t += 1
    print("Project actual costs: ", ACs[-1])
    mape, error = MAPE([ACs[-1]]*len(EAC_costs[:-1]), EAC_costs[:-1])
    print(f"Dynamic MAPE: {mape:.2f}")
    # plt.figure()
    # plt.plot(betas)
    # plt.xlabel("tracking periods")
    # plt.xticks(np.arange(len(betas)))
    # plt.ylabel("beta")
    # plt.savefig(f"figures/{data_file}-beta.png")
    return error, mape, ACs[-len(error):], EAC_costs[:-1], betas[:-1]

if __name__ == '__main__':
    if not os.path.isdir("figures"):
        os.makedirs("figures")
    fp = open(f"logs/costs/{data_file}.log", "w+")
    # fp.write(f"Dataset\tEVM\tStatic\tDynamic Recursive\tDynamic Without Recursive\n")
    fp.write(f"Dataset\tDynamic\n")

    error_evm, mape_evm = cost_forecasting_evm()
    # error_static, mape_dyn = cost_forecasting()
    # error_dyn_recursive, mape_without_recursive, acs, eacs_predict, betas = dynamic_cost_without_recursive()
    error_dyn, mape_dyn, acs, eacs_predict, betas = dynamic_cost()
    fp.write(f"{data_file}\t{mape_dyn:.2f}\n")
    # fp.write(f"{data_file}\t{mape_evm:.2f}\t{mape_static:.2f}\t{mape:.2f}\t{mape_without_recursive:.2f}\n")
    # fp.write(f"EVM: {mape_evm:.2f}\n")
    # fp.write(f"Static: {mape_static:.2f}\n")
    # fp.write(f"Dynamic: {mape:.2f}\n")

    # fig, (ax1, ax2) = plt.subplots(2 , 1 , sharex=False)
    fig = plt.figure()
    xs = np.arange(len(error_evm))
    plt.plot(xs, error_evm)
    # ax1.plot(xs, error_static)
    plt.plot(xs, error_dyn)
    plt.legend(["evm-cpi", "dynamic"], loc="upper left")
    plt.ylabel("MAPE (%)")

    # ax2.plot(xs, [acs[-1]]*len(eacs_predict))
    # ax2.plot(xs, eacs_predict)
    # ax2.legend(["actual EAC", "predict EAC"], loc="upper left")
    # ax3.plot(xs, betas)
    # # ax2.set_title("beta", y=1.08)
    plt.xticks(np.arange(len(error_evm)))
    plt.xlabel("Tracking periods")
    # plt.title("Dynamic MAPE = {:.3f}".format(mape), y=3.4)
    plt.savefig(f"figures/{data_file}.png")

