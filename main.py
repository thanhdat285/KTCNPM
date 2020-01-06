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
    "C2013-02": 0.140
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



def cost_forecasting():
    # planned duration
    BAC = baselines[0,2]
    # tracking periods
    tracking_periods = [x for x in sheetnames if "TP" in x]
    n_tracking_periods = len(tracking_periods)
    print("BAC:", BAC)
    print("Number of tracking periods:", n_tracking_periods)
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

        if EV < PV and not start_test:
            start_test = True
        if start_test:
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
    return error

# # Dynamic Beta

# In[10]:

def dynamic_cost():
    # planned duration
    BAC = baselines[0,2]
    # tracking periods
    tracking_periods = [x for x in sheetnames if "TP" in x]
    n_tracking_periods = len(tracking_periods)
    print("BAC:", BAC)
    print("Number of tracking periods:", n_tracking_periods)
    # init trend
    Ts_AC = [BAC/n_tracking_periods]
    Ts_EV = [BAC/n_tracking_periods]
    init_T = BAC/n_tracking_periods
    print("T0_AC = T0_EV: ", Ts_AC[0])

    def select_best_beta(cur_AC):
        betas = [] # list of tuples (beta, MAPE)
        for beta in np.arange(0.0, 1, 0.05):
            _ACs = [0]
            _Ts_AC = [0]
            predict_ACs = []
            for prev_period in range(0, cur_period):
                # predict AC of current period, cur_AC = prev_AC + trend_AC
                data_prev_period = dfs[tracking_periods[prev_period]].values[2:, cols]
                prev_AC = data_prev_period[0, 1]
                _ACs.append(prev_AC)
                _T_AC = beta*(_ACs[prev_period] - _ACs[prev_period-1]) + (1-beta) * _Ts_AC[prev_period-1]
                _Ts_AC.append(_T_AC)
                predict_AC = _ACs[prev_period-1] + (cur_period - prev_period)*_Ts_AC[prev_period-1]
                predict_ACs.append(predict_AC)
            error = MAPE([cur_AC]*len(predict_ACs), predict_ACs)
            betas.append((beta, error))
        # select best beta
        beta = sorted(betas, key=lambda x: x[1])[0][0]
        return beta

    def calculate_current_trend(ACs, beta):
        # ACs[0] must be equals to 0
        if len(ACs) == 1: return init_T
        prev_T = calculate_current_trend(ACs[:-1], beta)
        T = beta*(ACs[-1] - ACs[-2]) + (1-beta) * prev_T
        return T

    # Col 0 = ID, col 12 = Duration
    ACs = [0] # init AT0 = 0
    t = 1
    EVs = [0]
    EAC_costs = [] # predict project duration
    start_test = False
    cols = find_column_indices(dfs[tracking_periods[0]].values[1], ["ID", "Actual Cost", "Earned Value (EV)", "Planned Value (PV)"])
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
        T_AC = calculate_current_trend(ACs, beta)
        Ts_AC.append(T_AC)

        EV = data_period[0,2]
        PV = data_period[0,3]
        EVs.append(EV)
        T_EV = beta*(EVs[t] - EVs[t-1]) + (1-beta)*Ts_EV[t-1]
        Ts_EV.append(T_EV)

        if EV < PV and not start_test:
            start_test = True
        if start_test:
            k = (BAC-EVs[t]) / T_EV
            EAC = ACs[t] + k * T_AC
            EAC_costs.append(EAC)
            print(f"EAC: {EAC:.3f}\t{k}\t{T_AC}")
        # end calculate
        t += 1
    print("Project actual costs: ", ACs[-1])
    mape, error = MAPE([ACs[-1]]*len(EAC_costs[:-1]), EAC_costs[:-1])
    print(f"Dynamic MAPE: {mape:.2f}")
    # plt.plot(error)
    # plt.savefig(f"{data_file}-dyn.png")
    return error


if __name__ == '__main__':
    error_static = cost_forecasting()
    error_dyn = dynamic_cost()
    plt.plot(error_static)
    plt.plot(error_dyn)
    plt.legend(["static", "dynamic"], loc="upper right")
    plt.xlabel("Tracking periods")
    plt.ylabel("MAPE (%)")
    plt.savefig(f"{data_file}-cost.png")

