from scipy.stats import iqr
import numpy as np

class Utils:
    def fix_dataset_inconsistencies(dataframe, fill_value=None):
        dataframe = dataframe.replace([-np.inf, np.inf], np.nan)

        # This is done to avoid filling middle holes with backfilling.
        if fill_value is None:
            dataframe.iloc[0,:] = \
                dataframe.apply(lambda column: column.iloc[column.first_valid_index()], axis='index')
        else:
            dataframe.iloc[0,:] = \
                dataframe.iloc[0,:].fillna(fill_value)

        return dataframe.fillna(axis='index', method='pad').dropna(axis='columns')


    def estimate_outliers(data):
        return iqr(data) * 1.5

    def estimate_percent_gains(data, column='Close'):
        returns = Utils.get_returns(data, column=column)
        gains = Utils.estimate_outliers(returns)
        return gains

    def get_returns(data, column='Close'):
        return Utils.fix_dataset_inconsistencies(data[[column]].pct_change(), fill_value=0)

    def precalculate_ground_truths(data, column='Close', threshold=None):
        returns = Utils.get_returns(data, column=column)
        gains = Utils.estimate_outliers(returns) if threshold is None else threshold
        binary_gains = (returns[column] > gains).astype(int)
        return binary_gains

    def is_null(data):
        return data.isnull().sum().sum() > 0

    def is_sparse(data, column='Close'):
        n_bins = 10
        binary_gains = Utils.precalculate_ground_truths(data, column=column)
        bins = [n * (binary_gains.shape[0] // n_bins) for n in range(n_bins)]
        bins += [binary_gains.shape[0]]
        bins = [binary_gains.iloc[bins[n]:bins[n + 1]] for n in range(n_bins)]
        return all([bin.astype(bool).any() for bin in bins])

    def is_data_predictible(data, column):
        return not Utils.is_null(data) & Utils.is_sparse(data, column)
