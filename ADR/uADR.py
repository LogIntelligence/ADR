import sys

sys.path.append('..')
from ADR.preprocess import *


class uADR():
    def __init__(self, AN_ratio=0.05, nrows_per_sample=10, nrounds=1, highest_degree=1, add_parity=False, add_tf=True):
        self.AN_ratio = AN_ratio
        self.pN = 1 - self.AN_ratio
        self.highest_degree = highest_degree
        self.add_parity = add_parity
        self.add_tf = add_tf
        self.nrows_per_sample = nrows_per_sample
        self.nsamples_per_round = estimate_number_of_samples_per_round(self.AN_ratio, self.nrows_per_sample)
        self.nrounds = nrounds

    def run_one_round(self, x_train):
        x_validation = random_samples(x_train, min(self.nrows_per_sample * 10, x_train.shape[0]))
        extended_x_validation = extend_2darray(x_validation, highest_degree=1)

        n_sampled = 0
        while True:
            # x_sample, x_train = split_by_number(x_train, self.nrows_per_sample)
            x_sample = random_samples(x_train, self.nrows_per_sample)
            # print(x_sample)
            rank_x_sample = np.linalg.matrix_rank(x_sample)
            n_sampled += 1

            if n_sampled == 1:
                candidate = x_sample
                indicator1 = rank_x_sample
            else:
                if rank_x_sample < indicator1:
                    candidate = x_sample
                    indicator1 = rank_x_sample
                elif rank_x_sample == indicator1:
                    ns_extended_candidate = scipy.linalg.null_space(extend_2darray(candidate, highest_degree=1))
                    indicator2 = np.abs(np.dot(extended_x_validation, ns_extended_candidate)).mean()

                    ns_extended_x_sample = scipy.linalg.null_space(extend_2darray(x_sample, highest_degree=1))
                    dot_product_x_sample = np.abs(np.dot(extended_x_validation, ns_extended_x_sample)).mean()
                    if dot_product_x_sample < indicator2:
                        candidate = x_sample
                        indicator2 = dot_product_x_sample
                else:
                    pass

            if n_sampled > self.nsamples_per_round:
                break

        return candidate

    def run_multiple_rounds(self, x_train):
        collection_of_candidates = []
        for n_rounds in range(self.nrounds):
            collection_of_candidates.append(self.run_one_round(x_train))
        return collection_of_candidates

    def cross_check(self, collection_of_candidates):
        ns_collection_of_candidates = [scipy.linalg.null_space(i) for i in collection_of_candidates]
        num_candidates = len(collection_of_candidates)
        df_cross_check = pd.DataFrame()

        for i in range(num_candidates):
            for j in range(num_candidates):
                df_cross_check.loc[i, j] = validate_sample(collection_of_candidates[i], ns_collection_of_candidates[j],
                                                           1e-10)

        df_num_cross_check = df_cross_check.astype(int).sum(axis=1).values
        print(unique_count(df_num_cross_check))

        thre_numKilled = num_candidates - 1
        # print(idx_passed_candidates)
        hof = [collection_of_candidates[i] for i in range(num_candidates) if df_num_cross_check[i] < thre_numKilled]
        ns_hof = [ns_collection_of_candidates[i] for i in range(num_candidates) if
                  df_num_cross_check[i] < thre_numKilled]
        return hof, ns_hof

    def fit(self, x_train):
        self.collect_samples = self.run_multiple_rounds(x_train)
        self.collect_ns = [scipy.linalg.null_space(i) for i in self.collect_samples]

    def predict(self, x_test):
        # x_test = extend_2darray(x_test, highest_degree=self.highest_degree, add_parity=self.add_parity, add_tf=self.add_tf)
        collect_dot_prods_testing = [np.dot(x_test, ns) for ns in self.collect_ns]
        num_vios = np.array([(np.abs(i) > 1e-10).any(axis=1) for i in collect_dot_prods_testing])
        ratio_vios = num_vios.sum(axis=0) / num_vios.shape[0]
        prediction_y = ratio_vios >= self.pN
        return prediction_y

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return p_r_f(y_pred, y_test)
