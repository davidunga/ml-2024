from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
import numpy as np
import pandas as pd


class SDVBalancer:

    """ synthesize minority examples using SDV, to match number of majority """

    def __init__(self, kind: str = 'Gauss', shuff: bool = True):
        assert kind in ('Gauss', 'GAN')
        self.kind = kind
        self.shuff = shuff

    def fit_resample(self, X, y):

        orig_categories = {}
        for col in X.columns:
            if X[col].dtype == 'category':
                orig_categories[col] = X[col].cat.categories
                X[col] = X[col].astype(str)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(X)

        if self.kind == 'Gauss':
            sampler = GaussianCopulaSynthesizer(metadata=metadata)
        elif self.kind == 'GAN':
            sampler = CTGANSynthesizer(metadata=metadata)
        else:
            raise ValueError('Unknown sdv synthesizer kind')

        ycats = y.cat.categories
        assert len(ycats) == 2, "Currently supports binary targets only"

        pos_ycat, neg_ycat = ycats if 2 * np.sum(y == ycats[0]) < len(y) else ycats[::-1]
        ymask = y == pos_ycat

        positive_rows = np.nonzero(ymask)[0]
        imbalance_size = len(y) - 2 * len(positive_rows)

        sampler.fit(X.iloc[positive_rows])
        synthetic_positives = sampler.sample(num_rows=imbalance_size)

        X = pd.concat([X, synthetic_positives], axis=0).reset_index()
        ymask = np.r_[ymask, np.ones(len(synthetic_positives), bool)]

        if self.shuff:
            shuffled_ixs = np.random.default_rng(1).permutation(len(X))
            X = X.iloc[shuffled_ixs]
            ymask = ymask[shuffled_ixs]

        for col, orig_cat in orig_categories.items():
            X[col] = pd.Categorical(X[col], categories=orig_cat)

        y = pd.Series(ymask).map({True: pos_ycat, False: neg_ycat})

        return X, y
