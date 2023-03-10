import hdbscan
import numpy as np
import pandas as pd

from umap import UMAP
from typing import Union
from bertopic import BERTopic

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from spacy.lang.es.stop_words import STOP_WORDS as es_stop
from sklearn.feature_extraction.text import CountVectorizer

from .hdbscan_expt import HDBSCAN_MODEL


class TopicModeling:
    
    def __init__(
        self,
        excerpts: list,
        embeddings: Union[np.array, list],
    ):
        
        self.excerpts = excerpts
        self.embeddings = embeddings
        assert len(self.excerpts)==len(self.embeddings), ValueError("Excerts and Embeddings lenght must be equal")
        
    def _get_umap_model(self, 
                        n_neighbors: int = 10, 
                        n_compontens: int = 50, 
                        min_dist: float = 0.00, 
                        metric: str = "cosine", 
                        random_state: int = 42):
        
        assert len(self.excerpts)>n_compontens, ValueError("number of examples must be higher that number of components")
        
        return  UMAP(
            n_neighbors=n_neighbors,
            n_components=n_compontens,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
    
    
    def _find_best_score(self):
        self.hdbscan_score = self.hdb_class.calc_score()

    def _find_best_parameters(self):
        return self.hdb_class.get_best_params()
    
    def _get_hdbscan(self):
        
        self.hdb_class = HDBSCAN_MODEL(self.embeddings)
        self._find_best_score()
        params = self._find_best_parameters()
        
        self.hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                metric=params["metric"],
                cluster_selection_method=params["cluster_selection_method"],  # 'eom',
                prediction_data=True,
                min_samples=params["min_samples"]
        )
        return self.hdbscan_model
    
    
    def _get_stopwords(self):
        self.stopwords = list(fr_stop) + list(es_stop) + list(en_stop) + list(('nan',))
        return self.stopwords
    
    def _get_vectorizer(self, min_frequency: int = 5, n_grams: int = 2):
        
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, n_grams),
            #min_df=min_frequency,
            stop_words=self._get_stopwords()
        )
        
        return self.vectorizer_model
    
    def get_intertopic_dist_map(topic_model):
        return topic_model.visualize_topics()


    def get_topic_keyword_barcharts(topic_model, total_topics):
        return topic_model.visualize_barchart(
            top_n_topics=min(total_topics, 20),
            n_words=50,
            height=300,
            width=400)
    
    def get_reduced_embeddings(self, n_dimension: int = 2):
        
        _ = self._get_umap_model(n_compontens=n_dimension)
        return _.fit_transform(self.embeddings)

        
    def get_total_topics(
        self,
        language: str = "multilingual",
        low_memory: bool = True,
        verbose: bool = False,
        output_probabilites: bool = False,
        n_topics: Union[int, str] = "auto",
        topic_list: list = [],
        vectorizer_n_grams: int = 2,
        vectorizer_min_frequency: int = 5,
        umap_n_neighbors: int = 10, 
        umap_n_compontens: int = 50, 
        umap_min_dist: float = 0.00, 
        umap_metric: str = "cosine", 
        random_state: int = 42
    ):
        umap_model = self._get_umap_model(n_neighbors=umap_n_neighbors,
                                          n_compontens=umap_n_compontens,
                                          min_dist=umap_min_dist,
                                          metric=umap_metric,
                                          random_state=random_state)
        
        hdbscan_model = self._get_hdbscan()
        vectorizer = self._get_vectorizer(min_frequency=vectorizer_min_frequency, 
                                          n_grams=vectorizer_n_grams)
        
        self.model = BERTopic(
            language=language,
            low_memory=low_memory,
            umap_model=umap_model,
            verbose=verbose,
            calculate_probabilities=output_probabilites,
            nr_topics=n_topics,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            seed_topic_list=topic_list   
        )

        topics, probs = self.model.fit_transform(self.excerpts, embeddings=self.embeddings)
        self.general_topics_df  = self.model.get_document_info(self.excerpts)
        self.topics_df = pd.DataFrame({"topics": topics, "excerpts": self.excerpts})
          