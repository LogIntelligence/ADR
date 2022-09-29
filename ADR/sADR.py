from ADR.preprocess import *


class sADR():
    def __init__(self, hdegree=1, add_parity=False, add_tf=True):
        self.ns_x_train = None
        self.hdegree = hdegree
        self.add_parity = add_parity
        self.add_tf = add_tf

    def fit(self, x_train, y_train, sample=True):
        x_train, _ = split_to_N_AN(x_train, y_train)

        if sample:
            x_train = random_samples(x_train, min(x_train.shape[1] * 10, x_train.shape[0]))

        extended_x_train = extend_2darray(x_train, self.hdegree, self.add_parity, self.add_tf)
        # print(extended_x_train.shape)
        self.ns_x_train = scipy.linalg.null_space(extended_x_train)

        # if x_train.shape[0] > 10000:
        #     sample_x_train = x_train[0:10000, :]
        #     remain_x_train = x_train[10000:, :]
        #
        #     ns_sample_x_train = scipy.linalg.null_space(sample_x_train)
        #     dot_Remain_nsSample = np.dot(remain_x_train, ns_sample_x_train).__abs__()
        #     idx_valid = ((dot_Remain_nsSample > 1e-10).sum(axis=0)) < 1
        #     self.ns_x_train = ns_sample_x_train[:, idx_valid]
        # else:
        #     self.ns_x_train = scipy.linalg.null_space(x_train)

        # print(f'ns shape is {self.ns_x_train.shape}')
        return self.ns_x_train.shape[1]

    def predict(self, x):
        extended_x = extend_2darray(x, self.hdegree, self.add_parity, self.add_tf)
        # print(np.abs(np.dot(extended_x, self.ns_x_train)).shape)
        y_pred = (np.abs(np.dot(extended_x, self.ns_x_train)) > 1e-10).sum(axis=1) > 0
        return y_pred

    def predict_score(self, x):
        extended_x = extend_2darray(x, self.hdegree, self.add_parity, self.add_tf)
        y_pred_score = (np.abs(np.dot(extended_x, self.ns_x_train)) > 1e-10).sum(axis=1) / self.ns_x_train.shape[1]
        return y_pred_score

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        precision, recall, F = p_r_f(y_pred, y)
        return precision, recall, F
