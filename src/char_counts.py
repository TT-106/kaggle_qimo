"""Reuse per-document character counts while fitting vocabulary and IDF within each fold."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CharacterCounts:
    def __init__(self, text, article_ids, vectorizer_options):
        count_options = dict(vectorizer_options)
        self.weighting = {
            key: count_options.pop(key)
            for key in ("norm", "use_idf", "smooth_idf", "sublinear_tf")
            if key in count_options
        }
        self.min_df = count_options.pop("min_df", 1)
        self.max_df = count_options.pop("max_df", 1.0)
        self.max_features = count_options.pop("max_features", None)
        self.weight_dtype = count_options.pop("dtype", np.float64)
        self.article_ids = pd.Index(article_ids)
        if not self.article_ids.is_unique:
            raise ValueError("字符计数的文章 ID 有重复。")

        # Use a common column index for exact per-document counts. Selection,
        # document frequencies, and weights are computed from training rows below.
        counter = CountVectorizer(**count_options, dtype=np.int32)
        self.matrix = counter.fit_transform(text)
        self.term_names = counter.get_feature_names_out()

    def fit_fold(self, train_ids, validation_ids):
        train_positions = self.article_ids.get_indexer(train_ids)
        validation_positions = self.article_ids.get_indexer(validation_ids)
        if (train_positions < 0).any() or (validation_positions < 0).any():
            raise ValueError("字符计数缺少该折的文章。")
        training = self.matrix[train_positions]
        document_frequency = np.bincount(training.indices, minlength=training.shape[1])
        n_train = len(train_positions)
        low = self.min_df if isinstance(self.min_df, int) else self.min_df * n_train
        high = self.max_df if isinstance(self.max_df, int) else self.max_df * n_train
        available = document_frequency > 0
        columns = np.flatnonzero(available & (document_frequency >= low) & (document_frequency <= high))
        if not len(columns):
            raise ValueError("该训练折没有符合文档频次要求的字符片段。")
        if self.max_features is not None and len(columns) > self.max_features:
            occurrence_counts = np.asarray(training.sum(axis=0, dtype=self.weight_dtype)).ravel()[columns]
            kept = (-occurrence_counts).argsort()[:self.max_features]
            columns = np.sort(columns[kept])

        selected_train = training[:, columns].astype(self.weight_dtype)
        del training
        selected_validation = self.matrix[validation_positions][:, columns].astype(self.weight_dtype)
        weighting = TfidfTransformer(**self.weighting)
        weighting.fit(selected_train)
        return (
            weighting.transform(selected_train, copy=False),
            weighting.transform(selected_validation, copy=False),
            self.term_names[columns].tolist(),
        )
