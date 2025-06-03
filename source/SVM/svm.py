#%%
import pandas as pd
from collections import Counter
import os
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
"""データセットの準備※随時調整"""
df=pd.read_csv('../../database/drybeans_dataset.csv')
X_df=df.iloc[:,1:-1]
y_df=df.iloc[:,-1]
X =X_df.values
xname=X_df.columns.tolist()
y,yname= pd.factorize(y_df)#値を整数値にエンコード
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
outputfolder='../../output/simple_svm/adult/rbf1'
f1_score_average='binary'#f値の計測タイプ#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)
kernel='rbf' # カーネル: 'linear', 'rbf', 'poly', 'sigmoid' など
c=1#[0.1, 1, 10, 100],


# %%
"""Folder Creation"""
os.mkdir(outputfolder)
"""Model Training & Evaluate"""
datas=[]
for i, (train_idx, test_idx) in enumerate(kf.split(X_scaled,y)):
    """Train"""
    Xtr,Xte=X[train_idx],X[test_idx]#データをindexに応じて抽出
    ytr,yte=y[train_idx],y[test_idx]
    svm = SVC(kernel=kernel,class_weight='balanced') 
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
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# StratifiedKFoldのインスタンスを作成
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model = SVC(kernel='rbf', C=1)
# 交差検証を実施
scores = cross_val_score(model, X, y, cv=skf)

print("各分割でのスコア:", scores)
print("平均スコア:", scores.mean())
# %%
