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
"""データセットの準備※随時調整"""
df=pd.read_csv('../../database/adult_dataset.csv')
X_df=df.iloc[:,1:-1]
y_df=df.iloc[:,-1]
X =X_df.values
xn=X_df.columns.tolist()
y,yn= pd.factorize(y_df)#値を整数値にエンコード
y,yn =y_df.values, [0,1]
print('table_shape;',X_df.shape)
print('class sample;',y_df.value_counts())
#%%
"""Parameter Set"""
outputfolder='../../output/simple_cart/adult/ho067_dp4-11_dn3-7_2'
k=5#繰り返し回数
n=500#創る決定木の数
train_data_ratio=0.4
tr_ratio_range=[0.3,0.7]
depth=20
depth_range=[4,11]
f1_score_average='weighted'#f値の計測タイプ#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)
#%%
"""Folder Creation"""
os.mkdir(outputfolder)
"""Model Training & Evaluate"""
for i in range(k):
    """Data Split"""
    Xtr,Xte,ytr,yte = train_test_split(X, y, train_size=0.67, random_state=i,stratify=y)#データ分割
    datas=[]
    imp_datas=[]
    dtinfos=[]  
    for j in range(n):  
        """Train"""
        #Xuse,Xnonuse,yuse,ynonuse = train_test_split(Xtr, ytr, train_size=train_data_ratio,random_state=j,stratify=ytr)
        Xuse,Xnonuse,yuse,ynonuse = train_test_split(Xtr, ytr, train_size=np.random.uniform(tr_ratio_range[0],tr_ratio_range[1]),random_state=j,stratify=ytr)
        #clf=DecisionTreeClassifier(max_depth=depth, class_weight='balanced',splitter='best',random_state=0)#モデル作成，呼び出し
        clf=DecisionTreeClassifier(max_depth=np.random.randint(depth_range[0],depth_range[1]+1), class_weight='balanced',splitter='best',random_state=0)#モデル作成，呼び出し
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
        #dtinfo=DTinfo()
        #dtinfo.read_clf(clf,xn,yn)
        #f_average=dtinfo.f_average
    
        data={'AC(mk)':round(ac,3), 'AC(mk_all)':round(ac2,3), 'AC(test)':round(ac3,3), 'size':size,
            'F1(mk)':round(fm,3), 'F1(mk_all)':round(fm2,3), 'F1(test)':round(fm3,3), 
            #'faverage':f_average, 
            'importance_uniformity':round(imp_uniformity,3)}
        datas.append(data)
        imp_data=dict(zip(xn, importances))
        imp_datas.append(imp_data)
        #dtinfos.append(dtinfo)
    """Output Model Information"""
    values_list=[]
    columns=list(datas[0].keys())+['feature_importance']+list(imp_datas[0].keys())
    for j in range(n):
        values=list(datas[j].values())+[None]+list(imp_datas[j].values())
        values_list.append(values)
    os.mkdir(outputfolder+'/simpleCART_clfs_run'+str(i))
    #for j in range(n):
    #    dtinfos[i].write_csv(outputfolder+'/simpleCART_clfs_run{}/clf{}.csv'.format(str(i),str(j)))
    df =pd.DataFrame(values_list,columns=columns)
    print('run{}:ave...{}'.format(i,df.mean()))
    df.to_csv(outputfolder+'/simpleCART_data_run'+str(i)+'.csv')
# %%
"""making graph"""
import matplotlib.pyplot as plt
import seaborn as sns

y='size'
x='F1(test)'
for i in range(k):
    df=pd.read_csv(outputfolder+'/simpleCART_data_run'+str(i)+'.csv')
    fig=plt.figure(figsize=(8,5))
    ax=sns.scatterplot(data=df,x=x,y=y,alpha=0.75,palette="Set1")
    fig.savefig(outputfolder+'/graph_run'+str(i)+'')
    plt.close(fig)
    

# %%
