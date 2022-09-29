# %%
import sys

sys.path.append("..")

import itertools
import pandas as pd
import scipy.linalg
import re
from scipy.special import expit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
# from loglizer import preprocessing
# import loglizer.preprocessing
import numpy as np
import os
from tqdm import tqdm_notebook as tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


def split_to_N_AN(x, y):
    y_bool = y >= 1
    x_AN = x[y_bool]
    x_N = x[~y_bool]
    return x_N, x_AN


def split_to_train_test_by_ratio(x, y, train_ratio=0.5):
    x, y = np.array(x), np.array(y)
    num_rows, num_cols = x.shape
    num_train = np.int(num_rows * train_ratio)
    shuffled_indices = np.arange(num_rows)
    np.random.shuffle(shuffled_indices)
    idx_train = shuffled_indices[0:num_train]
    idx_test = shuffled_indices[num_train:]

    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]
    return x_train, y_train, x_test, y_test


def split_to_train_test_by_num(x, y, num_train):
    x, y = np.array(x), np.array(y)
    num_rows, num_clos = x.shape
    shuffled_indices = np.arange(num_rows)
    np.random.shuffle(shuffled_indices)

    idx_train = shuffled_indices[0:num_train]
    idx_test = shuffled_indices[num_train:]

    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]
    return x_train, y_train, x_test, y_test


def split_to_windows_by_time(para, raw_data, event_mapping_data, dict_label, col_label, datetime_format, col_datetime):
    """ split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = para['save_path'] + 'sliding_' + str(para['window_size']) + 'h_' + str(
        para['step_size']) + 'h.csv'

    # =============divide into sliding windows=========#
    start_end_index_list = []  # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    label_data = raw_data[:, col_label]

    if datetime_format == "timestamp":
        timestamp_data = raw_data[:, col_datetime]
    elif datetime_format == "dateANDtime":
        datetime_data = " ".join([raw_data[:, col_datetime[i]] for i in col_datetime])

    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = timestamp_data[0]
        # print(start_time)
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in timestamp_data:
            if cur_time < start_time + para['window_size'] * 3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair = tuple((start_index, end_index))
                start_end_index_list.append(start_end_pair)
                break

        # print(f"start_end_index_list is {start_end_index_list}")

        # move the start and end index until next sliding window
        while end_index < log_size:
            start_time = start_time + para['step_size'] * 3600
            end_time = end_time + para['step_size'] * 3600
            for i in range(start_index, end_index):
                if timestamp_data[i] < start_time:
                    i += 1
                else:
                    break
            for j in range(end_index, log_size):
                if timestamp_data[j] < end_time:
                    j += 1
                else:
                    break
            start_index = i
            end_index = j
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        # print(f"start_end_index_list is {start_end_index_list}")
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset\n' % inst_number)
        # np.savetxt(sliding_file_path, start_end_index_list, delimiter=',', fmt='%d')
    else:
        print('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list = []
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)
    # for e in expanded_indexes_list:
    #     print(e)
    # event_mapping_data = [row[0] for row in event_mapping_data]
    # print(event_mapping_data)
    eventID_list = set(event_mapping_data.values())
    event_num = len(eventID_list)
    print('There are %d log events' % event_num)

    # =============get labels and event count of each sliding window =========#
    labels = []
    event_count_matrix = np.zeros((inst_number, event_num))
    print(event_count_matrix.shape)
    for j in range(inst_number):
        label = 0  # 0 represent success, 1 represent failure
        # print(f"expanded_indexes_list[j] is {expanded_indexes_list[j]}")
        for k in expanded_indexes_list[j]:
            # print(f"k is {k}")
            event_index = event_mapping_data[k]
            event_count_matrix[j, event_index] += 1
            # print(f"label_data[k] is {label_data[k]}")
            # if not label_data[k]:
            #     print("label is 0")
            if list(dict_label.keys()) == ["Normal"]:
                if label_data[k] == None:
                    print("label is none")
                if label_data[k] not in dict_label["Nojurmal"]:
                    # print(f"label = 1")
                    label = 1
                    continue
            elif list(dict_label.keys()) == "Anomaly":
                if label_data[k] in dict_label["Anomaly"]:
                    # print(f"label = 1")
                    label = 1
                    continue

        labels.append(label)
        # print(sum(event_count_matrix[j]))
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies" % sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    return event_count_matrix, labels


def event_matrix_count(row):
    ecm_iden_dict = pd.DataFrame(row["seq_EventId"]).value_counts().to_dict()
    ecm_iden_df = pd.DataFrame(ecm_iden_dict, index=[row.name])
    return ecm_iden_df


def event_sequence_by_identifier(df_logs, col_identifier, col_EventId, col_bLabel):
    df_seq = {}  # df_seq = pd.DataFrame(columns=['bLabel', 'seq_EventId', 'seq_LineId', 'seq_bLabel'])
    df_seq_ecm = pd.DataFrame()
    for idx, row in tqdm(df_logs.to_dict('index').items(), total=df_logs.shape[0]):
        iden = row[col_identifier]
        if iden not in df_seq.keys():
            df_seq[iden] = {}
            df_seq[iden]['bLabel'] = False
            df_seq[iden]['seq_EventId'] = []
            df_seq[iden]['seq_LineId'] = []
            df_seq[iden]['seq_bLabel'] = []
        if col_bLabel is not None:
            df_seq[iden]['bLabel'] = df_seq[iden][col_bLabel] or row[col_bLabel]
            df_seq[iden]['seq_bLabel'].append(row[col_bLabel])
        df_seq[iden]['seq_EventId'].append(row[col_EventId])
        df_seq[iden]['seq_LineId'].append(row['LineId'])
    df_seq = pd.DataFrame.from_dict(df_seq, orient='index')
    # print(df_seq.head(5))
    df_seq_ecm = df_seq.parallel_apply(event_matrix_count, axis=1)
    df_seq_ecm = pd.concat(df_seq_ecm.tolist())
    # for idx, row in tqdm(df_seq.to_dict('index').items(), total=df_seq.shape[0]):
    #     # print(row['seq_EventId'])
    #     ecm_iden_dict = pd.DataFrame(row["seq_EventId"]).value_counts().to_dict()
    #     ecm_iden_df = pd.DataFrame(ecm_iden_dict, index=[idx])
    #     df_seq_ecm = pd.concat([df_seq_ecm, ecm_iden_df])

    # for iden, df_iden in tqdm(df_logs.groupby(col_identifier)):
        # print(iden)
        # df_iden['Timestamp'] = df_iden['Timestamp'].astype(int)
        # df_iden = df_iden.sort_values('Timestamp')
        # df_seq.loc[iden, 'seq_EventId'] = df_iden['EventId'].to_list()
        # df_seq.loc[iden, 'seq_LineId'] = df_iden['LineId'].to_list()
        # df_seq.loc[iden, 'seq_bLabel'] = df_iden['bLabel'].to_list()
        # ecm_iden_dict = df_iden['EventId'].value_counts().to_dict()
        # ecm_iden_df = pd.DataFrame(ecm_iden_dict, index=[iden])
        # df_seq_ecm = pd.concat([df_seq_ecm, ecm_iden_df])
        #
        # df_seq.loc[iden, 'bLabel'] = int(any(df_iden[col_bLabel]))

    return df_seq, df_seq_ecm


def event_count_by_identifier(df_logs, col_identifier, col_EventId, col_bLabel=None, list_events_ids=None):
    if col_bLabel is not None:
        event_count_by_iden = pd.DataFrame(columns=["bLabel"])
    else:
        event_count_by_iden = pd.DataFrame()

    if list_events_ids is not None:
        cols = list_events_ids + event_count_by_iden.columns.to_list()
        event_count_by_iden = pd.DataFrame(columns=cols)

    for idx, log in df_logs.iterrows():
        if not log[col_identifier] in event_count_by_iden.index:
            event_count_by_iden.loc[log[col_identifier], :] = 0
        if not log[col_EventId] in event_count_by_iden.columns:
            event_count_by_iden[log[col_EventId]] = 0

        event_count_by_iden.loc[log[col_identifier], log[col_EventId]] += 1

        if col_bLabel is not None:
            if int(log[col_bLabel]) == 1:
                event_count_by_iden.loc[log[col_identifier], "bLabel"] = 1

    if col_bLabel is not None:
        Labels = event_count_by_iden['bLabel']
        df_ECM = event_count_by_iden.drop('bLabel', axis=1)
        return df_ECM, Labels
    else:
        return event_count_by_iden


def mean_max_min(a):
    print(f"mean is {np.mean(a)}")
    print(f"max is {np.amax(a)}")
    print(f"min is {np.amin(a)}")


def stats_x_y(x, y, hdegree=1, add_parity=False, add_Exist=False):
    print(f"x's shape is {x.shape}")
    x = extend_2darray(x, highest_degree=hdegree, add_parity=add_parity, add_tf=add_Exist)
    x_N, x_AN = split_to_N_AN(x, y)
    if x_N.shape[0] > 100000:
        x_N = random_samples(x_N, 1000)
    if x_AN.shape[0] > 100000:
        x_AN = random_samples(x_AN, 1000)

    print(f"x_N's shape is {x_N.shape}")
    print(f"x_AN's shape is {x_AN.shape}")
    print(f'x_N rank is {np.linalg.matrix_rank(x_N)}')
    print(f'x_AN rank is {np.linalg.matrix_rank(x_AN)}')

    # print(f"x's rank is {np.linalg.matrix_rank(x)}")
    # print(f"x's rank is {scipy.linalg.interpolative.estimate_rank(x.astype(float), 0.1)}")

    # if x_N.shape[0] > 0:
    #     print(f"x_N's rank is {np.linalg.matrix_rank(x_N)}")
    # if x_AN.shape[0] > 0:
    #     print(f"x_AN's rank is {np.linalg.matrix_rank(x_AN)}")

    if x_N.shape[0] > 0:
        N_null_space = scipy.linalg.null_space(x_N)
    if x_AN.shape[0] > 0:
        AN_null_space = scipy.linalg.null_space(x_AN)
    # if x_N.shape[0] > 0:
    #     N_thresholds = np.abs(np.dot(x_N, N_null_space)).max()
    # if x_AN.shape[0] > 0:
    #     AN_thresholds = np.abs(np.dot(x_AN, AN_null_space)).max()

    print("-----N dot ns(AN)-----")
    if x_AN.shape[0] > 0:
        print(unique_count((np.abs(np.dot(x_N, AN_null_space)) > 1e-10).any(axis=1)))
        N_dot_for_nv = ((np.abs(np.dot(x, AN_null_space)) > 1e-10).sum(axis=0)) / x_N.shape[0]
        # print('min, max, mean:')
        # print(N_dot_for_nv.min(), N_dot_for_nv.max(), N_dot_for_nv.mean())
    print("-----AN dot ns(N)-----")
    if x_N.shape[0] > 0:
        print(unique_count((np.abs(np.dot(x_AN, N_null_space)) > 1e-10).any(axis=1)))
        AN_dot_for_nv = ((np.abs(np.dot(x, N_null_space)) > 1e-10).sum(axis=0)) / x_AN.shape[0]
        # print('min, max, mean:')
        # print(AN_dot_for_nv.min(), AN_dot_for_nv.max(), AN_dot_for_nv.mean())


def unique_count(a):
    return np.array(np.unique(a, return_counts=True)).T


def add_precedence(npa_windows, num_precede, labels=None):
    num_rows = npa_windows.shape[0]

    npa_windows_with_precede = npa_windows[num_precede:, :]

    current_precede = 1
    while current_precede <= num_precede:
        windows_current_precede = npa_windows[num_precede - current_precede:num_rows - current_precede, :]
        npa_windows_with_precede = np.concatenate((windows_current_precede, npa_windows_with_precede), axis=1)
        current_precede += 1

    if labels == None:
        return npa_windows_with_precede, labels[num_precede:]
    else:
        return npa_windows_with_precede


def split_by_timewindows(df_logs, col_timestamp, windows_size, step_size):
    num_rows = df_logs.shape[0]
    # =============divide into sliding windows=========#
    start_end_index_list = []  # list of tuples, tuple contains two number, which represent the start and end of sliding time window

    timestamp_data = df_logs[col_timestamp].astype(int)

    # split into sliding window
    start_time = timestamp_data[0]
    print(f"first timestamp is {start_time}")
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in timestamp_data:
        if cur_time < start_time + windows_size * 3600:
            end_index += 1
            end_time = cur_time
        else:
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
            break

    # move the start and end index until next sliding window
    while end_index < num_rows:
        start_time = start_time + step_size * 3600
        end_time = end_time + step_size * 3600
        for i in range(start_index, end_index):
            if timestamp_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, num_rows):
            if timestamp_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j
        start_end_pair = tuple((start_index, end_index))
        start_end_index_list.append(start_end_pair)
    # print(f"start_end_index_list is {start_end_index_list}")
    windows_number = len(start_end_index_list)
    print('there are %d sliding windows in this dataset\n' % windows_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list = []
    for t in range(windows_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(windows_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    assert len(expanded_indexes_list) == windows_number

    return expanded_indexes_list


def event_count_by_time_window(df_logs, col_timestamp, col_EventId, window_size, step_size, col_Label=None,
                               keep_N_AN="All"):
    if keep_N_AN == "All":
        pass
    elif keep_N_AN == "N":
        df_logs = df_logs.drop(df_logs[df_logs[col_Label] == 1].index)
    elif keep_N_AN == "AN":
        df_logs = df_logs.drop(df_logs[df_logs[col_Label] == 0].index)

    if col_Label != None:
        event_count_by_windows = pd.DataFrame(columns=["bLabel"])
    else:
        event_count_by_windows = pd.DataFrame()

    windows_rows_list = split_by_timewindows(df_logs, col_timestamp, window_size, step_size)

    for idx_window, rows_list in enumerate(windows_rows_list):
        if not event_count_by_windows.index.contains(idx_window):
            event_count_by_windows.loc[idx_window, :] = 0
        for idx_row in rows_list:
            curEventId = df_logs.loc[idx_row, col_EventId]
            if not event_count_by_windows.columns.contains(curEventId):
                event_count_by_windows[curEventId] = 0

            event_count_by_windows.loc[idx_window, curEventId] += 1
            if col_Label != None:
                if int(df_logs.loc[idx_row, col_Label]) >= 1:
                    event_count_by_windows.loc[idx_window, "bLabel"] = 1

    return event_count_by_windows


def ECM_by_NumEventWindow(df_logs, list_EventIds, col_timestamp, col_EventId, window_size, step_size, col_bLabel):
    df_logs = df_logs.sort_values(by=[col_timestamp])
    num_rows = df_logs.shape[0]
    ECM_by_windows = pd.DataFrame(columns=list_EventIds + ["bLabel"])

    idx_window = 0
    start_index = 0
    end_index = 0 + window_size
    if window_size > num_rows:
        raise Exception("window size greater than num_rows")

    while end_index <= num_rows:
        if not idx_window in ECM_by_windows.index:
            ECM_by_windows.loc[idx_window, :] = 0
        for idx_row in list(range(start_index, end_index)):
            curEventId = df_logs.loc[idx_row, col_EventId]
            if not curEventId in ECM_by_windows.columns:
                ECM_by_windows[curEventId] = 0

            ECM_by_windows.loc[idx_window, curEventId] += 1
            if int(df_logs.loc[idx_row, col_bLabel]) >= 1:
                ECM_by_windows.loc[idx_window, "bLabel"] = 1
        idx_window += 1
        start_index += step_size
        end_index += step_size

    bLabels = ECM_by_windows['bLabel']
    df_ECM_by_windows = ECM_by_windows.drop('bLabel', axis=1)
    return df_ECM_by_windows, bLabels


# def load_hdfs_npz_to_ECM(log_path):
#     if log_path.endswith("npz"):
#         data = np.load(log_path, allow_pickle=True)
#         x = data['x_data']
#         feature_extractor = preprocessing.FeatureExtractor()
#         x = feature_extractor.fit_transform(x)
#         y = data['y_data']
#         return x, y

def load_hdfs_structured_logs(log_path, labels_path=None):
    df_logs = pd.read_csv(log_path, sep=',', header=0)
    print("extracting block_id...")
    df_logs["BlockId"] = df_logs.parallel_apply(lambda row: re.findall(r'blk_-?\d+', row['Content'])[0], axis=1)
    if labels_path is not None:
        df_labels = pd.read_csv(labels_path, sep=',', header=0, index_col=0)
        df_logs_bLabels = df_logs.parallel_apply(lambda row: 1 if df_labels.loc[row['BlockId'], 'Label'] == "Anomaly" else 0,
                                        axis=1).values
        df_logs["bLabel"] = df_logs_bLabels
        return df_logs
    return df_logs


def load_bgl_structured_logs(log_path):
    df_logs = pd.read_csv(log_path, sep=',', header=0)
    df_logs_blabels = df_logs.apply(lambda row: 0 if row['Label'] == '-' else 1, axis=1).values
    return df_logs, df_logs_blabels


def transform(X, term_weighting=None, normalization=None, oov=False, min_count=1):
    num_instance, num_event = X.shape
    if term_weighting == 'tf-idf':
        df_vec = np.sum(X > 0, axis=0)
        idf_vec = np.log(num_instance / (df_vec + 1e-8))
        idf_matrix = X * np.tile(idf_vec, (num_instance, 1))
        X = idf_matrix
    if normalization == 'zero-mean':
        mean_vec = X.mean(axis=0)
        mean_vec = mean_vec.reshape(1, num_event)
        X = X - np.tile(mean_vec, (num_instance, 1))
    elif normalization == 'sigmoid':
        X[X != 0] = expit(X[X != 0])
    return X


def comb(vector, highest_degree):
    vector = list(vector)
    comb_vector = [1] + vector
    degree = 2
    while degree < highest_degree + 1:
        comb_vector_with_degree = [i for i in itertools.combinations_with_replacement(vector, degree)]
        comb_vector = comb_vector + comb_vector_with_degree
        degree += 1
    return comb_vector


def comb_product(vector, highest_degree=2):
    comb_list = comb(vector, highest_degree)
    # print(comb_list)
    # print([np.prod(e) for e in comb_list])
    return [np.prod(e) for e in comb_list]


# def add_bias(a):
#     try:
#         (num_row, num_col) = a.shape
#         bias = np.ones(shape=(a.shape[0],1))
#         return np.concatenate((bias, a), axis=1)
#     except:
#         return np.concatenate(([1], a))

# def extend_2darray(a, highest_degree, add_parity=False, add_tf=False):
#     comb_product_2d = []
#     for vector in a:
#         comb_product_2d.append(comb_product(vector, highest_degree))
#     comb = np.array(comb_product_2d)
#
#     if add_parity:
#         parity_a = np.remainder(a, 2)
#         comb = np.concatenate((comb, parity_a), axis=1)
#
#     if add_tf:
#         exist_a = a > 0
#         comb = np.concatenate((comb, exist_a), axis=1)
#     return comb

def extend_2darray(a, highest_degree, add_parity=False, add_tf=False):
    a = np.array(a)
    if highest_degree == 0:
        comb = a
    elif highest_degree == 1:
        cons_terms = np.ones((a.shape[0], 1))
        comb = np.concatenate((cons_terms, a), axis=1)
    else:
        comb = np.apply_along_axis(lambda x: comb_product(x, highest_degree), 1, a)

    if add_parity:
        parity_a = np.remainder(a, 2)
        comb = np.concatenate((comb, parity_a), axis=1)

    if add_tf:
        exist_a = a > 0
        comb = np.concatenate((comb, exist_a), axis=1)
    return comb


def get_num_cols_extend_2darray(a, highest_degree, add_parity=False, add_tf=False):
    a = a[0:1, :]
    comb = extend_2darray(a, highest_degree, add_parity, add_tf)
    return comb.shape[1]


# a = np.array([[1,2,3],[3,4,5]])
# print(comb_product_2d(a,2))
# x = [1,2,3,4]
# comb_x = comb_product(x, 1)
# print(comb_x)

def p_r_f_mcc(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    # print("precision, recall, F1:")
    # print(precision, recall, f1)
    return precision, recall, f1, mcc


def p_r_f(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    # mcc = matthews_corrcoef(y_true, y_pred)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    # print("precision, recall, F1:")
    # print(precision, recall, f1)
    return precision, recall, f1


def prob_Nsample(num_P, AN_ratio, sample_ratio):
    num_AN = int(num_P * AN_ratio)
    num_N = num_P - num_AN
    num_sample = int(num_P * sample_ratio)
    #     print(num_sample)
    num_C_P = scipy.special.comb(num_P, num_sample)
    #     print(num_C_P)
    num_C_N = scipy.special.comb(num_N, num_sample)
    #     print(num_C_N)
    return num_C_N / num_C_P


def get_number_of_sample(num_P, AN_ratio, sample_ratio=0.05):
    num_sample = 1
    while True:
        prob_AllN = prob_Nsample(num_P, AN_ratio, sample_ratio)
        prob_AtLeast1AN = 1 - prob_AllN
        prob_AtLeast1AN_for_num_samples = 1 - prob_AtLeast1AN ** num_sample
        if prob_AtLeast1AN_for_num_samples > 0.999:
            print(f"sample {num_sample} times, success rate is {prob_AtLeast1AN_for_num_samples}")
            return num_sample
        num_sample += 1


def estimate_number_of_samples_per_round(AN_ratio, nrows_per_sample):
    nsamples_per_round = 1
    while True:
        # print(nsamples_per_round)
        p_N = 1 - AN_ratio
        p_type1 = p_N ** nrows_per_sample
        p_none_type1 = 1 - p_type1
        p_at_least_1_type1_for_nsamples = 1 - p_none_type1 ** nsamples_per_round
        if p_at_least_1_type1_for_nsamples > 0.9999:
            # print(f"sample {nsamples_per_round} times, success rate is {prob_AtLeast1AN_for_nsamples}")
            return nsamples_per_round
        nsamples_per_round += 1


def random_samples(x_train, n_samples, y_train=None):
    n_rows = x_train.shape[0]
    idx = np.random.randint(n_rows, size=n_samples)

    rand_samples = x_train[idx, :]
    if y_train is None:
        return rand_samples
    else:
        y_rand_samples = y_train[idx, :]
        return rand_samples, y_rand_samples


def repeat_samples(x, no_samples):
    n_rows = x.shape[0]
    rand_int = np.random.randint(n_rows)
    sample = x[rand_int:rand_int + 1, :]

    for i in range(1, no_samples):
        rand_int = np.random.randint(n_rows)
        new_sample = x[rand_int:rand_int + 1, :]
        sample = np.concatenate((sample, new_sample), axis=0)

    return sample


def random_split_by_number(x, no_part1, y):
    no_rows = x.shape[0]
    shuffled_indices = np.arange(no_rows)
    np.random.shuffle(shuffled_indices)
    idx_part1 = shuffled_indices[0:no_part1]
    idx_part2 = shuffled_indices[no_part1:]
    x_part1 = x[idx_part1, :]
    x_part2 = x[idx_part2, :]

    if y is None:
        return x_part1, x_part2
    else:
        y_part1 = y[idx_part1]
        y_part2 = y[idx_part2]
        return x_part1, y_part1, x_part2, y_part2


def validate_sample(validate_sample, null_space, thre=1e-10):
    dot_prod = np.dot(validate_sample, null_space).__abs__()
    return dot_prod.max() > thre


def stats(a):
    return np.mean(a), np.amax(a), np.amin(a), np.sum(a)


def eval_by_nullspace(x_test, y_test, nullspace, hdegree, add_parity, add_tf):
    x_test = extend_2darray(x_test, hdegree, add_parity=add_parity, add_tf=add_tf)
    dot_prod = np.dot(x_test, nullspace).__abs__()
    y_pred = (dot_prod > 1e-10).sum(axis=1) >= 1
    print("precision, recall, F, MCC is:")
    print(p_r_f_mcc(y_pred, y_test))


def eval_sample(train, test, test_y):
    ns_train = scipy.linalg.null_space(train)
    dot_prod = np.dot(test, ns_train).__abs__()
    y_predict = (dot_prod > 1e-10).sum(axis=1) >= 1
    p, r, f1, mcc = p_r_f_mcc(y_predict, test_y)
    return (p, r, f1, mcc)
