#%%
import pandas as pd
from collections import Counter
import os
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
"""データセットの準備※随時調整"""
df=pd.read_csv('../../database/strokes_dataset.csv')
X_df=df.iloc[:,1:-1]
y_df=df.iloc[:,-1]
X =X_df.values
xname=X_df.columns.tolist()
y,yname= pd.factorize(y_df)#値を整数値にエンコード
y,yname=y_df.values,[0,1]
print('table_shape;',X_df.shape)
print('class sample;',y_df.value_counts())
"""データのk分割"""
k=10
kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
for fold, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    print("train;",Counter(y[train_idx]))
    print("test;",Counter(y[test_idx]))
print(df.columns)
"""#%%
#データ分布確認
import seaborn as sns
import matplotlib.pyplot as plt
for x in X_df.columns:
    sns.histplot(data=df,x=x,hue='Outcome')
    plt.show()"""
    
#%%
"""Parameter Set"""
outputfolder='../../output/simple_svm/strokes/poly_default'
f1_score_average='binary'#f値の計測タイプ#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)
kernel='poly' # カーネル: 'linear', 'rbf', 'poly', 'sigmoid' など
# %%
"""Folder Creation"""
os.mkdir(outputfolder)
"""Model Training & Evaluate"""
datas=[]
for i, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    """Scaling and Training"""
    Xtr,Xte=X[train_idx],X[test_idx]#データをindexに応じて抽出
    scaler = StandardScaler()#データを正規化（特徴量スケーリング）
    Xtr_scaled = scaler.fit_transform(Xtr)#トレーニングデータに対して正規化を行っている？※分割前に正規化はいけない。テストデータに合わさって正規化されることになりカンニングになる
    Xte_scaled = scaler.transform(Xte)#トレーニングデータのパラメータでテストデータを正規化している？
    ytr,yte=y[train_idx],y[test_idx]
    svm = SVC(kernel=kernel,class_weight='balanced') 
    svm.fit(Xtr_scaled, ytr)
    """Evaluate"""
    ac   = accuracy_score(ytr,svm.predict(Xtr_scaled))#学習データでの精度
    ac3  = accuracy_score(yte,svm.predict(Xte_scaled))#がちテストデータでの精度
    fm  = f1_score(ytr,svm.predict(Xtr_scaled),average=f1_score_average)#学習データでのf値
    fm3 = f1_score(yte,svm.predict(Xte_scaled),average=f1_score_average)#がちテストデータでのf値

    
    data={'AC(mk)':round(ac,3), 'AC(test)':round(ac3,3),
          'F1(mk)':round(fm,3), 'F1(test)':round(fm3,3)}
    datas.append(data)

"""Output Model Information"""
df = pd.DataFrame(datas)
print(df.mean())
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
