from ADR.preprocess import *
import timeit


class ADR_Online:
    def __init__(self, window_size, highest_degree=1, add_parity=False, add_tf=True):
        self.windows_size = window_size
        self.highest_degree = highest_degree
        self.add_parity = add_parity
        self.add_tf = add_tf

    def fit(self, train_df_logs, list_EventIds):
        self.list_EventIds = list_EventIds
        collect_df_windows = pd.DataFrame()
        collect_bLabels = pd.Series()
        col_identifier = 'Node'
        window_size = self.windows_size
        step_size = 1

        for iden, df_iden in train_df_logs.groupby(col_identifier):
            if iden == "-":
                continue

            if df_iden.shape[0] < window_size:
                continue

            df_iden_label = df_iden["bLabel"].astype(int).any()
            if df_iden_label == 1:
                continue

            df_iden = df_iden.reset_index(drop=True)
            ECM_windows, ECM_bLabels = ECM_by_NumEventWindow(df_iden, list_EventIds=self.list_EventIds, col_timestamp="Timestamp", col_EventId="EventId", window_size=window_size, step_size=step_size, col_bLabel="bLabel")

            collect_df_windows = collect_df_windows.append(ECM_windows, sort=False)

        self.df_ECM = collect_df_windows
        npa_ECM = collect_df_windows.values
        self.extended_ECM = extend_2darray(npa_ECM, highest_degree=self.highest_degree, add_parity=self.add_parity, add_tf=self.add_tf)
        self.ns_extended_ECM = scipy.linalg.null_space(self.extended_ECM)

    def evaluate(self, test_df_logs):
        print('grouping logs to windows...')
        num_rows = test_df_logs.shape[0]
        collect_df_windows = pd.DataFrame()
        collect_bLabels = pd.Series()
        col_identifier = 'Node'
        t0 = timeit.default_timer()
        window_size = self.windows_size
        step_size = 1
        for iden, df_iden in test_df_logs.groupby(col_identifier):
            if iden == "-":
                continue
            if df_iden.shape[0] < window_size:
                continue
            df_iden = df_iden.reset_index(drop=True)
            ECM_windows, ECM_bLabels = ECM_by_NumEventWindow(df_iden, list_EventIds=self.list_EventIds, col_timestamp="Timestamp", col_EventId="EventId", window_size=window_size, step_size=step_size, col_bLabel="bLabel")

            collect_df_windows = collect_df_windows.append(ECM_windows, sort=False)
            collect_bLabels = collect_bLabels.append(ECM_bLabels)

        print('anomaly detection...')
        npa_bLabels = collect_bLabels.values
        npa_event_count = collect_df_windows.values

        extended_npa_event_count = extend_2darray(npa_event_count, highest_degree=self.highest_degree, add_parity=self.add_parity, add_tf=self.add_tf)
        dot_prod = np.dot(extended_npa_event_count, self.ns_extended_ECM).__abs__()
        y_pred = (dot_prod > 1e-10).sum(axis=1) >= 1
        precision, recall, F, mcc = p_r_f_mcc(y_pred, npa_bLabels)
        print("precision, recall, F, mcc is:")
        print((precision, recall, F, mcc))
        return precision, recall, F, mcc

