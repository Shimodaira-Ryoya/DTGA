#%%
"""Import Library"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from module.importance_deal import importance_deal,importance_uniformity
from module.dt_info import DTinfo
"""Load Dataset"""
pas='../../database/breast_cancer_dataset.csv'
df=pd.read_csv(pas)
X=df.iloc[:,1:-1]
y=df.iloc[:,-1]
xname=X.columns.tolist()
yname=[0,1]
print(X.shape)
print(y.value_counts())
"""Data Split"""
Xtr,Xte,ytr,yte = train_test_split(X.to_numpy(), y.to_numpy(), 
                                   train_size=0.6, random_state=1,stratify=y)#データ分割
#%%
"""Parameter Set"""
outputfolder='../../output/simple_cart/test_pima_tr0.6'
num=500#創る決定木の数
train_range=[0.7,1.0]#トレーニングデータ採用割合
depth_range=[3,12]#決定木深さ
f1_score_average='binary'#f値の計測タイプ#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)

#%%
"""Folder Creation"""
os.mkdir(outputfolder)
"""Model Training & Evaluate"""
datas=[]
imp_datas=[]
dtinfos=[]
for i in range(num):
    """Train"""
    Xuse,Xnonuse,yuse,ynonuse = train_test_split(Xtr, ytr, train_size=np.random.uniform(train_range[0],train_range[1]))
    clf=DecisionTreeClassifier(max_depth=np.random.randint(depth_range[0],depth_range[1]),
                               class_weight='balanced',splitter='best',random_state=0)#モデル作成，呼び出し
    clf.fit(Xuse,yuse)
    
    """Evaluate"""
    importances = clf.feature_importances_
    imp_uniformity = importance_uniformity(importances)
    ac   = accuracy_score(yuse,clf.predict(Xuse))#学習データでの精度
    ac2  = accuracy_score(ytr,clf.predict(Xtr))#評価データでの精度
    ac3  = accuracy_score(yte,clf.predict(Xte))#がちテストデータでの精度
    fm  = f1_score(yuse,clf.predict(Xuse),average=f1_score_average)#学習データでのf値
    fm2 = f1_score(ytr,clf.predict(Xtr),average=f1_score_average)#評価データでのf値
    fm3 = f1_score(yte,clf.predict(Xte),average=f1_score_average)#がちテストデータでのf値
    size=clf.tree_.node_count
    dtinfo=DTinfo()
    dtinfo.read_clf(clf,xname,yname)
    f_average=dtinfo.f_average
    
    data={'AC(mk)':round(ac,3), 'AC(mk_all)':round(ac2,3), 'AC(test)':round(ac3,3), 'size':size,
          'F1(mk)':round(fm,3), 'F1(mk_all)':round(fm2,3), 'F1(test)':round(fm3,3), 
          'faverage':f_average, 'importance_uniformity':round(imp_uniformity,3)}
    datas.append(data)
    
    imp_data=dict(zip(xname, importances))
    imp_datas.append(imp_data)
    dtinfos.append(dtinfo)
"""Output Model Information"""
values_list=[]
columns=list(datas[0].keys())+['feature_importance']+list(imp_datas[0].keys())
for i in range(num):
    values=list(datas[i].values())+[None]+list(imp_datas[i].values())
    values_list.append(values)

os.mkdir(outputfolder+'/simpleCART_clfs')
for i in range(num):
    dtinfos[i].write_csv(outputfolder+'/simpleCART_clfs/clf{}.csv'.format(str(i)))

df =pd.DataFrame(values_list,columns=columns)
df.to_csv(outputfolder+'/simpleCART_data.csv')