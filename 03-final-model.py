
import json
import re
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import base
import pandas as pd
from sklearn.pipeline import FeatureUnion
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
import gensim

df  = pd.read_pickle("dummy")


class MeanEmbeddingVectorizer(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 100

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):
        return X[self.col_names]
        
        # Return an array with the same number of rows as X and one
        # column for each in self.col_namesa



class GroupbyEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, column, estimator_factory):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators
        self.column = column
        self.est = {}
        self.estimator_factory = estimator_factory
    
    def fit(self, X, y):
        grouped = X.groupby(self.column)
        for name, ind in grouped.groups.items():
            self.est[name] = self.estimator_factory().fit(X.loc[ind], y.loc[ind])
        return self

    def predict(self, X):
        # Call the appropriate predict method for each row of X
        grouped = X.groupby(self.column)
        res = pd.DataFrame(index=X.index, columns=['pred'])
        for name, ind in grouped.groups.items():
            res.loc[ind, 'pred'] = self.est[name].predict_proba(X.loc[ind])[:,1]
        return res.values.reshape(-1)
    
    def transform(self, X):
        # Call the appropriate predict method for each row of X
        grouped = X.groupby(self.column)
        res = pd.DataFrame(index=X.index, columns=['pred'])
        for name, ind in grouped.groups.items():
            res.loc[ind, 'pred'] = self.est[name].predict_proba(X.loc[ind])[:,1]
        return res.values.reshape(-1, 1)



class NameEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, estimator):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators

        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        # Call the appropriate predict method for each row of X

        return self.estimator.predict_proba(X)[:,1].reshape(-1,1)
    
    def transform(self, X):
        # Call the appropriate predict method for each row of X

        return self.estimator.predict_proba(X)[:,1].reshape(-1,1)






def category_factory():
    pipe = Pipeline([
    ('cst', ColumnSelectTransformer(['bump_price'])),
    ('lr', LogisticRegression())
    ])
    return pipe

def designer_factory():
    pipe = Pipeline([
    ('cst', ColumnSelectTransformer(['bump_price'])),
    ('lr', LogisticRegression())
    ])
    return pipe



class PriceTransformer(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array(X['bump_price']).reshape(-1, 1)
    
class NameTransformer(base.BaseEstimator, base.TransformerMixin):
 # We will need these in transform()
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    def transform(self, X):
        return list(X['name'])
class DesignerTransformer(base.BaseEstimator, base.TransformerMixin):
 # We will need these in transform()
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    def transform(self, X):
        return list(X['designer'])


class ProbaTransformer(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, estimator):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators

        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        # Call the appropriate predict method for each row of X

        return self.estimator.predict_proba(X)[:,1].reshape(-1,1)
    
    def transform(self, X):
        # Call the appropriate predict method for each row of X

        return self.estimator.predict_proba(X)[:,1].reshape(-1,1)


def logistic_factory():
    w2v_pipe = Pipeline([
    ('name', NameSplitTransformer()),
    ('w2v', MeanEmbeddingVectorizer(w2v))
    ])


names = [x.lower().split(" ") for x in list(df['name'])]
model = gensim.models.Word2Vec(names, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.vectors))


w2v_price = FeatureUnion([
    ("word2vec", w2v_pipe),
    ("price", PriceTransformer())
])

etree_combined = Pipeline([
    ("word2vec vectorizer", w2v_price),
("logit", LogisticRegression())])
return etree_combined


w2v_pipe = Pipeline([
    ('name', NameSplitTransformer()),
    ('w2v', MeanEmbeddingVectorizer(w2v))
])

w2v_price = FeatureUnion([
    ("word2vec", w2v_pipe),
    ("price", PriceTransformer())
])

logit_combined = Pipeline([
    ('w2v_price', w2v_price),
    ("logistic", ProbaTransformer(LogisticRegression()))
])


price_pipe = Pipeline(steps=[
    ('pt', PriceTransformer()),
    ('scale', MinMaxScaler())
])
name_price = FeatureUnion([
    ('name',name_pipe),
    ('price', price_pipe)
])
name_price_est = Pipeline([
    ('features', name_price),
    ('xgbrg', XGBClassifier(objective='binary:logistic', n_estimators=5, max_depth = 9))
])


name_model = ProbaTransformer(name_price_est)
designer_model = GroupbyEstimator('designer',  designer_factory)
category_model = GroupbyEstimator('category', category_factory)
full_logit_w2v = logit_combined
designer_logit_w2v = GroupbyEstimator('designer', logistic_factory)
category_logit_w2v = GroupbyEstimator('category', logistic_factory)


X_train, X_test, y_train, y_test = train_test_split(df, df['sold'], test_size=0.1, random_state=42)


ensemble = FeatureUnion([
    ('designer', designer_model),
    ('category', category_model),
    ('name', name_model),
    ("w2v_designer", designer_logit_w2v),
    ("w2v_category", category_logit_w2v),
    ("w2v_full", full_logit_w2v)
])
final_model = Pipeline(steps=[
    ('ensemble', ensemble),
    ('lr', ProbaTransformer(LogisticRegression()))
])


name_model.fit(X_train, y_train)
category_model.fit(X_train, y_train)
designer_model.fit(X_train, y_train)
full_logit_wv2
designer_logit_w2v
category_logit_w2v
final_model.fit(X_train, y_train)


models = {
    "name":name_model,
    "designer":designer_model,
    "category":category_model,
    "w2v_full":full_logit_w2v,
    "w2v_designer":designer_logit_w2v,
    "w2v_category":category_logit_w2v,
    "full ensemble":final_model
}


losses = []
for name, model in models.items():
    loss = log_loss(y_test, model.transform(X_test))
    losses.append((name, loss))
    print("{}: {}".format(name, loss))
losses = sorted(losses, key= lambda el:-el[1])




fig = plt.figure(figsize = (12,8))
for count, item in enumerate(losses):
    plt.bar(count, item[1], tick_label=item[0])

plt.xticks(range(0,7),[x[0] for x in losses], rotation=25, fontsize=20)
plt.ylabel("Log Loss", fontsize = 25)
fig.savefig("loss.png", bbox_inches='tight')



import dill
dill.settings['recurse']=True
dill_file = open("ensemble", "wb")
dill_file.write(dill.dumps(final_model))
dill_file.close()

