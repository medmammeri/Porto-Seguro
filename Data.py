import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv',na_values=-1)
test = pd.read_csv('data/test.csv',na_values=-1)

cat_feat = train.filter(like='cat', axis=1).columns.values.tolist()
bin_feat = train.filter(like='bin', axis=1).columns.values.tolist()
num_feat = list(set(list(train)) - set(cat_feat)- set(bin_feat))
num_feat.remove('target')
num_feat.remove('id')
features_names = cat_feat + bin_feat + num_feat

for var in bin_feat + cat_feat:
    train[var] = train[var].astype('category')
    test[var] = test[var].astype('category')
    
Train_cnt = train[num_feat]
Train_bin = train[bin_feat]
Train_cat = pd.get_dummies(train[cat_feat],
                              prefix_sep='#',
                              drop_first=True)
Test_cnt = test[num_feat]
Test_bin = test[bin_feat]
Test_cat = pd.get_dummies(test[cat_feat],
                              prefix_sep='#',
                              drop_first=True)

X_cnt = Train_cnt.as_matrix()
X_cat = Train_cat.as_matrix()
X_bin = Train_bin.as_matrix()

X_cnt_test = Test_cnt.as_matrix()
X_cat_test = Test_cat.as_matrix()
X_bin_test = Test_bin.as_matrix()

X = np.concatenate((X_cnt, X_bin, X_cat), axis=1)
X_test = np.concatenate((X_cnt_test, X_bin_test, X_cat_test), axis=1)
y = train['target']

np.save('/tmp/X', X)
np.save('/tmp/X_test', X_test)
np.save('/tmp/y', y)