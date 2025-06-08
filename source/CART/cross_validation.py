#%%
"""ライブラリのインポート"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from module.importance_deal import importance_deal,importance_uniformity
from module.dt_info import DTinfo
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
"""データセットの準備※随時調整"""
df=pd.read_csv('../../database/digits_dataset.csv')
X_df=df.iloc[:,1:-1]
y_df=df.iloc[:,-1]
X =X_df.values
xname=X_df.columns.tolist()
y,yname= pd.factorize(y_df)#値を整数値にエンコード
#y,yname=y_df.values,[0,1]
print('table_shape;',X_df.shape)
print('class sample;',y_df.value_counts())
"""データのk分割"""
k=10
kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
for fold, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    print("train;",Counter(y[train_idx]))
    print("test;",Counter(y[test_idx]))
    
#%%
"""Parameter Set"""
outputfolder='../../output/simple_cart/digits/depth13'
f1_score_average='macro'#f値の計測タイプ#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)
depth=13

#%%
"""Folder Creation"""
os.mkdir(outputfolder)
"""Model Training & Evaluate"""
datas=[]
imp_datas=[]
dtinfos=[]
for i, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    """Train"""
    Xtr,Xte=X[train_idx],X[test_idx]#データをindexに応じて抽出
    ytr,yte=y[train_idx],y[test_idx]
    clf=DecisionTreeClassifier(max_depth=depth,
                               class_weight='balanced',splitter='best',random_state=0)#モデル作成，呼び出し
    clf.fit(Xtr,ytr)
    """Evaluate"""
    importances = clf.feature_importances_
    imp_uniformity = importance_uniformity(importances)
    ac   = accuracy_score(ytr,clf.predict(Xtr))#学習データでの精度
    ac3  = accuracy_score(yte,clf.predict(Xte))#がちテストデータでの精度
    fm  = f1_score(ytr,clf.predict(Xtr),average=f1_score_average)#学習データでのf値
    fm3 = f1_score(yte,clf.predict(Xte),average=f1_score_average)#がちテストデータでのf値
    size=clf.tree_.node_count
    dtinfo=DTinfo()
    dtinfo.read_clf(clf,xname,yname)
    f_average=dtinfo.f_average
    
    data={'AC(mk)':round(ac,3), 'AC(test)':round(ac3,3), 'size':size,
          'F1(mk)':round(fm,3), 'F1(test)':round(fm3,3), 
          'faverage':f_average, 'importance_uniformity':round(imp_uniformity,3)}
    datas.append(data)
    
    imp_data=dict(zip(xname, importances))
    imp_datas.append(imp_data)
    dtinfos.append(dtinfo)
"""Output Model Information"""
values_list=[]
columns=list(datas[0].keys())+['feature_importance']+list(imp_datas[0].keys())
for i in range(k):
    values=list(datas[i].values())+[None]+list(imp_datas[i].values())
    values_list.append(values)

os.mkdir(outputfolder+'/simpleCART_clfs')
for i in range(k):
    dtinfos[i].write_csv(outputfolder+'/simpleCART_clfs/clf{}.csv'.format(str(i)))

df =pd.DataFrame(values_list,columns=columns)
print(df.mean())
df.to_csv(outputfolder+'/simpleCART_data.csv')
d# %%
