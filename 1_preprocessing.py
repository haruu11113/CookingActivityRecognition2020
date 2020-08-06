#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# In[2]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '1_preprocessing.ipynb'])


# In[4]:


# '''
# データの処理に必要なツールをインポートする
# ''
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
import japanize_matplotlib

# DataFrameを表示する際の折り返す横幅の設定
pd.set_option('display.width', 200)

# DataFrameを表示する際、全てのDataFrame を表示させる設定
# デフォルトは一番最後のDataFrame一つのみ
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# DataFrameを表示するとき、表示する行数
# とりあえず最大５０００行表示するようにする
pd.set_option('display.max_rows', 5000)

# DataFrameを省略せず全部表示（多分ね）
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_columns', 100)


# In[108]:


right_arm_file_list = glob("train/right_arm/*")
df = pd.read_csv(right_arm_file_list[3])
df['timestamp'] = pd.to_datetime(df['timestamp'])
# df.dtypes
# df.sort_values('timestamp')


# In[101]:


f = open('train/labels.txt')
label_file = f.read()  # ファイル終端まで全て読んだデータを返す
f.close()
# labels_list = [s.split(',') for s in label_file.split('\n')] # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
# labels = np.array(labels_list)
LABELS = pd.DataFrame([s.split(',', 2) for s in label_file.split('\n')], columns=['file_name', 'food', 'activity'])
LABELS = LABELS.dropna()


# In[156]:


features_list = []
for i, file in enumerate(LABELS['file_name']):
    df = pd.read_csv('./train/right_arm/{0}.csv'.format(file))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    feat = [LABELS.loc[i, 'activity'], df['X'].min(), df['X'].max(), df['X'].mean(), df['X'].max()-df['X'].min(),]
    features_list.append(feat)

FEATURES = pd.DataFrame(features_list)


# ## Takeにラベルつけ

# In[254]:


take_file_list = LABELS[LABELS['activity'] == 'Take,']
# FEATURES['label'] = FEATURES[0]
FEATURES['label'] = '0'
# FEATURES.loc[FEATURES[0].str.contains('Take,'), 'label']='1'
FEATURES.loc[FEATURES[0] == 'Take,', 'label']='2'
FEATURES['label'].value_counts()


# ## classification

# In[255]:


FEATURES.columns


# In[256]:


X_train,X_test,Y_train,Y_test = train_test_split(FEATURES[[1, 2, 3, 4]], FEATURES['label'], test_size=0.3, random_state=1)


# In[260]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model_ml = RandomForestClassifier(n_estimators=4,n_jobs=-1, criterion= 'gini', max_depth=2, random_state=15)

model_ml.fit(X_train,Y_train)
Y_predict = model_ml.predict(X_test)

print(classification_report(Y_test,Y_predict))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from tqdm import tqdm\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import f1_score\n\nmax_score = 0\nSearchMethod = 0\nRFC_grid = {RandomForestClassifier(): {"n_estimators": [i for i in range(1, 500)],\n                                       "criterion": ["gini", "entropy"],\n                                       "max_depth":[i for i in range(1, 500)],\n                                       "random_state": [15],\n                                      }}\n\n#ランダムフォレストの実行\nfor model, param in tqdm(RFC_grid.items()):\n    clf = GridSearchCV(model, param)\n    clf.fit(X_train, Y_train)\n    Y_predict = clf.predict(X_test)\n    score = f1_score(Y_test, Y_predict, average="micro")\n\n    if max_score < score:\n        max_score = score\n        best_param = clf.best_params_\n        best_model = model.__class__.__name__\n\nprint("ベストスコア:{}".format(max_score))\nprint("モデル:{}".format(best_model))\nprint("パラメーター:{}".format(best_param))')


# In[ ]:


LABELS['activity'].value_counts()


# In[ ]:





# In[ ]:




