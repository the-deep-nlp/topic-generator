import logging
import hdbscan
import numpy as np
import pandas as pd
from umap import UMAP
from typing import Union, Optional
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

from sklearn.feature_extraction.text import CountVectorizer

from topic_generator.stopwords_fr import STOP_WORDS_FR as fr_stop
from topic_generator.stopwords_en import STOP_WORDS_EN as en_stop
from topic_generator.stopwords_es import STOP_WORDS_ES as es_stop
from topic_generator.hdbscan_expt import HDBSCANModel

logging.getLogger().setLevel(logging.INFO)

class TopicGenerator:
    """ Generates the topics from the excerpts """
    def __init__(
        self,
        excerpts: list,
        embeddings: Union[np.array, list],
    ):
        self.excerpts = excerpts
        self.embeddings = embeddings

        self.hdb_class = None
        self.general_topics_df = pd.DataFrame([])

        assert len(self.excerpts)==len(self.embeddings), \
            ValueError("Excerts and Embeddings lenght must be equal")

    def _get_umap_model(
        self,
        n_neighbors: int=10,
        n_components: int=50,
        min_dist: float=0.00,
        metric: str="cosine",
        random_state: int=42
    ):
        assert len(self.excerpts)>n_components, \
            ValueError("number of examples must be higher that number of components")
        return UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )

    def _find_best_score(self):
        self.hdbscan_score = self.hdb_class.calc_score()

    def _find_best_parameters(self):
        return self.hdb_class.get_best_params()

    def _get_hdbscan(self):
        """ Generates a hdbscan model by searching parameters """
        self.hdb_class = HDBSCANModel(self.embeddings)
        self._find_best_score()
        params = self._find_best_parameters()
        logging.info("The best parameters for hdbscan are: %s", params)
        return hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            metric=params["metric"],
            cluster_selection_method=params["cluster_selection_method"],
            prediction_data=True,
            min_samples=params["min_samples"]
        )

    def _get_stopwords(self):
        """ Gets the stop words from different languages """
        return list(fr_stop) + list(es_stop) + list(en_stop) + list(('nan',))

    def _get_vectorizer(self, min_frequency: int=5, n_grams: int=2):
        """ Gets the vectorizer """
        return CountVectorizer(
            ngram_range=(1, n_grams),
            #min_df=min_frequency,
            stop_words=self._get_stopwords()
        )

    def get_reduced_embeddings(self, n_dimension: int=2):
        """ Reduces the vector dimensions to the specified one """
        umap_model = self._get_umap_model(n_components=n_dimension)
        return umap_model.fit_transform(self.embeddings)

    def get_total_topics(
        self,
        language: str="multilingual",
        low_memory: bool=True,
        verbose: bool=False,
        output_probabilites: bool=False,
        n_topics: Union[int, str]="auto",
        topic_list: Optional[list]=None,
        vectorizer_n_grams: int=2,
        vectorizer_min_frequency: int=5,
        umap_n_neighbors: int=10,
        umap_n_components: int=50,
        umap_min_dist: float=0.20,
        umap_metric: str="cosine",
        diversity: float=0.5,
        random_state: int=42
    ):
        """ Gets all the possible topics from excerpts in a dataframe """
        umap_model = self._get_umap_model(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state
        )

        hdbscan_model = self._get_hdbscan()
        vectorizer = self._get_vectorizer(
            min_frequency=vectorizer_min_frequency,
            n_grams=vectorizer_n_grams
        )
        representation_model = MaximalMarginalRelevance(diversity=diversity)

        topic_model = BERTopic(
            language=language,
            low_memory=low_memory,
            umap_model=umap_model,
            verbose=verbose,
            calculate_probabilities=output_probabilites,
            nr_topics=n_topics,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            seed_topic_list=topic_list,
            representation_model=representation_model
        )

        topic_model.fit_transform(self.excerpts, embeddings=self.embeddings)
        self.general_topics_df  = topic_model.get_document_info(self.excerpts)
