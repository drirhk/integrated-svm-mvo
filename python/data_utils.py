import pandas as pd
import pandas_datareader as pdr
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time


def AV(start, end, freq='weekly', n_y=None, use_cache=False, save_results=False,
       AV_key=None, get_factors=False, tick_list=None):
    """Load data from alpha vantage
    Inputs
    start: start date
    end: end date
    split: train-validation-test split as percentages
    freq: data frequency (daily, weekly, monthly)
    use_cache: Boolean. State whether to load cached data or download data
    save_results: Boolean. State whether the data should be cached for future use.
    Outputs
    X: TrainTest object with feature data split into train, validation and test subsets
    Y: TrainTest object with asset data split into train, validation and test subsets
    """

    if use_cache:
        X = pd.read_pickle('./cache/asset_' + freq + '.pkl')
        print("cache")
        data_result = (X[:-1], None)

    if get_factors and use_cache:
        F = pd.read_pickle('./cache/factors_' + freq + '.pkl')
        data_result = (X[:-1], F[:-1])
    if use_cache is False:
        if tick_list is None:
            tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT',
                         'JNJ',
                         'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']

        if n_y is not None:
            tick_list = tick_list[:n_y]

        if AV_key is None and use_cache is False:
            print("""A personal AlphaVantage API key is required to load the asset pricing data. If you do not have a 
            key, you can get one from www.alphavantage.co (free for academic users)""")
            AV_key = input("Enter your AlphaVantage API key: ")

        ts = TimeSeries(key=AV_key, output_format='pandas', indexing_type='date')

        # Download asset data
        X = []
        updated_list = []
        for tick in tick_list:
            if freq == "daily":
                data, _ = ts.get_daily_adjusted(symbol=tick)
            elif freq == "weekly":
                try:
                    data, _ = ts.get_weekly_adjusted(symbol=tick)
                except:
                    print("ticker ", tick, " Invalid")
                    data = None
            if data is not None:
                data = data['5. adjusted close']
                X.append(data)
                updated_list.append(tick)
                time.sleep(1.0)
        X = pd.concat(X, axis=1)
        X = X[::-1]
        print(X)
        X = X['1999-1-1':end].pct_change()
        print(X)
        X = X[start:end]
        X.columns = updated_list
        data_result = (X[:-1],)

    if get_factors:
        # Download factor data
        dl_freq = '_weekly'
        f = pdr.get_data_famafrench('F-F_Research_Data_Factors' + dl_freq, start=start,
                                    end=end)[0]
        rf_df = f['RF']
        f = f.drop(['RF'], axis=1)
        #         mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+dl_freq, start=start, end=end)[0]
        #         st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+dl_freq, start=start, end=end)[0]
        #         lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+dl_freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        ##F = pd.concat([F, mom_df, st_df, lt_df], axis=1) / 100
        f = pd.concat([f], axis=1) / 100

        data_result = (X[:-1], f[:-1])

    if save_results:
        X.to_pickle('./cache/asset_' + freq + '.pkl')
    if get_factors and save_results:
        f.to_pickle('./cache/factors_' + freq + '.pkl')

    return data_result


def form_vectors(X, n_obs, valid_dates):
    """Forms returns vectors to make covariance calculations easier
    """
    if valid_dates is None:
        valid_dates = X.index[n_obs:]
    data_to_append = []
    for date in valid_dates:
        # get the last n_obs
        last_n = X[X.index < date].iloc[-n_obs:]
        last_n.index.names = ['vector_date']
        last_n['date'] = date
        last_n.reset_index(inplace=True)
        last_n = last_n.set_index(['date', 'vector_date'])
        data_to_append.append(last_n)
    return pd.concat(data_to_append)


def geo_sum_overflow(iterable):
    iterable2 = iterable.drop(iterable.columns[iterable.isna().any(axis=0)], axis=1)
    # print(len(iterable2))
    # print(len(iterable2.columns))
    return np.exp(np.log(iterable2).sum())


def geo_mean_overflow(iterable):
    # we allow some missing values since we are estimating returns
    iterable2 = iterable.drop(iterable.columns[iterable.isna().sum(axis=0) > 5], axis=1)
    # print(len(iterable2))
    # print(len(iterable2.columns))
    return np.exp(np.log(iterable2).mean())


def fundamentals_AV(tick_list, valid_dates, use_cache=False, save_results=False, key='W8WIGW27G58KMF82', period=52,
                    interval='weekly'):
    "get the fundamentals data for the SVM"
    ti = TechIndicators(key=key, output_format='pandas')
    X = []
    i = 0
    if use_cache:
        X = pd.read_pickle('./cache/attributes_' + interval + '.pkl')
    else:
        for tick in tick_list:
            # get the metrics we require
            if i % 50 == 0:
                print("iteration ", i)
            i += 1
            try:
                data_willr, meta_data = ti.get_willr(symbol=tick, interval=interval, time_period=period)
                time.sleep(0.8)
                data_adxr, meta_data = ti.get_adxr(symbol=tick, interval=interval, time_period=period)
                time.sleep(0.8)
                data_cci, meta_data = ti.get_cci(symbol=tick, interval=interval, time_period=period)
                time.sleep(0.8)
                data_aroon, meta_data = ti.get_aroon(symbol=tick, interval=interval, time_period=period)
                time.sleep(0.8)
                data_mfi, meta_data = ti.get_mfi(symbol=tick, interval=interval, time_period=period)
                time.sleep(0.8)
                data_ultosc, meta_data = ti.get_ultosc(symbol=tick, interval=interval)
                time.sleep(0.8)
                data_dx, meta_data = ti.get_dx(symbol=tick, interval=interval, time_period=period)
                time.sleep(0.8)
                X_tick = pd.concat([data_willr, data_adxr, data_cci, data_aroon, data_mfi, data_ultosc, data_dx],
                                   axis=1)
                # form the 1 week look back vector
                X_tick_qtr = form_vectors(X_tick, n_obs=1, valid_dates=valid_dates)

                tuples = [(tick, date, vector_date) for (date, vector_date) in X_tick_qtr.index]

                X_tick_qtr.index = pd.MultiIndex.from_tuples(tuples, names=["ticker", "date", "vector_date"])
                X.append(X_tick_qtr)
            except:
                print("ticker ", tick, " Invalid")

        X = pd.concat(X, axis=0)

    if save_results:
        X.to_pickle('./cache/attributes_' + interval + '.pkl')
    return X


class Instance:
    def __init__(self, mean, covariance, out_of_sample_ret, asset_attributes):
        self.mean = mean
        self.covariance = covariance
        self.out_of_sample_ret = out_of_sample_ret
        self.asset_attributes = asset_attributes
        self.trade_date = None
        self.estimation_end_date = None
        self.qtr = None
        self.ticker_list = None

    @property
    def check_dimensions(self):
        n_mean = len(self.mean)
        n_cov = len(self.covariance)
        n_attributes = len(self.asset_attributes)
        n_oot = len(self.asset_attributes)
        if n_mean == n_cov == n_attributes == n_oot:
            print("success")


def extract_parameters(instance, N=None, asset_attributes = None):
    """
    extract the parameters for the instance
    :param instance:
    :param N:
    :return:
    """
    cov_, mean_ = instance.covariance.values, instance.mean.values
    if N is None:
        N = len(mean_)
    if asset_attributes is None:
        asset_attributes = ["Momentum", "Vol"]
    cov = cov_[:N, :N]
    mean = np.expand_dims(mean_[:N], axis = 1)
    tics = instance.ticker_list
    # get the wharton research data for the valid tickers for the month
    # restrict the wharton research data to the columns of interest
    if asset_attributes != "all":
        Y = instance.asset_attributes.loc[:, asset_attributes]  # Y matrix in formulation
    else:
        Y = instance.asset_attributes
    oot_return = instance.out_of_sample_ret

    return mean, cov, tics, Y, oot_return
