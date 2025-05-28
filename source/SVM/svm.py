#%%
import pandas as pd
from collections import Counter
import os
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
"""データセットの準備※随時調整"""
df=pd.read_csv('../../database/strokes_dataset.csv')
X=df.iloc[:,1:-1]
y=df.iloc[:,-1]
xname=X.columns.tolist()
yname=[0,1]
X=X.values
print(X.shape)
print(y.value_counts())
y=y.values
"""データのk分割"""
k=10
kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
for fold, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    print("train;",Counter(y[train_idx]))
    print("test;",Counter(y[test_idx]))
    
#%%
"""Parameter Set"""
outputfolder='../../output/simple_svm/strokes/strokes_CV10_rbf_c1'
f1_score_average='binary'#f値の計測タイプ#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)
kernel='rbf' # カーネル: 'linear', 'rbf', 'poly', 'sigmoid' など
c=1#[0.1, 1, 10, 100],


# %%
"""Folder Creation"""
os.mkdir(outputfolder)
"""Model Training & Evaluate"""
datas=[]
for i, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    """Train"""
    Xtr,Xte=X[train_idx],X[test_idx]#データをindexに応じて抽出
    ytr,yte=y[train_idx],y[test_idx]
    svm = SVC(kernel=kernel,C=c,class_weight='balanced') 
    svm.fit(Xtr, ytr)
    """Evaluate"""
    ac   = accuracy_score(ytr,svm.predict(Xtr))#学習データでの精度
    ac3  = accuracy_score(yte,svm.predict(Xte))#がちテストデータでの精度
    fm  = f1_score(ytr,svm.predict(Xtr),average=f1_score_average)#学習データでのf値
    fm3 = f1_score(yte,svm.predict(Xte),average=f1_score_average)#がちテストデータでのf値

    
    data={'AC(mk)':round(ac,3), 'AC(test)':round(ac3,3),
          'F1(mk)':round(fm,3), 'F1(test)':round(fm3,3)}
    datas.append(data)

"""Output Model Information"""
df = pd.DataFrame(datas)
df.to_csv(outputfolder+'/simpleSVM_data.csv')
# %%
