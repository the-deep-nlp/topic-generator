import hdbscan
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


class HDBSCANModel:
    """ Creates a hdbscan model """
    def __init__(self, embeddings):
        self.hdbscan_model = hdbscan.HDBSCAN(
            gen_min_span_tree=True
        ).fit(embeddings)

        self.params_dist = {
            "min_cluster_size": [3, 4, 5, 6, 7, 10, 25, 50],
            "min_samples": [2, 3],
            "cluster_selection_method": ["leaf", "eom"],
            "metric": ["euclidean", "manhattan"]
        }
        self.random_search = None
        self.embeddings = embeddings
        self.iter_search = 10
        self.seed = 1234

    def calc_score(self):
        """ Calculates the validity scores by searching parameters """
        score = make_scorer(
            hdbscan.validity.validity_index,
            greater_is_better=True
        )
        self.random_search = RandomizedSearchCV(
            self.hdbscan_model,
            param_distributions=self.params_dist,
            n_iter=self.iter_search,
            scoring=score,
            random_state=self.seed
        )
        self.random_search.fit(self.embeddings)
        # Return dbcv score
        return self.random_search.best_estimator_.relative_validity_

    def get_best_params(self):
        """ Gets the best hdbscan parameters """
        return self.random_search.best_params_
