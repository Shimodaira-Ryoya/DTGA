#%%
"""ライブラリのインポート"""
import os
import numpy as np
import pandas as pd
import timeit
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from module.problem2 import Problem
from module.nsgaii import nsgaii
from module import ea_base as ea
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
"""データセットの準備※随時調整"""
df=pd.read_csv('../../database/adult_dataset.csv')
X=df.iloc[:,1:-1]
y=df.iloc[:,-1]
xname=X.columns.tolist()
yname=[0,1]
#yname=["SEKER" ,"BARBUNYA" , "BOMBAY" , "CALI" ,"HOROZ" , "SIRA" ,"DERMASON"]
print(X.columns)
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
# %%
"""パラメータ設定※随時調整"""
output_folder='../../output/method2/strokes/dn05_nonuse_fmsz_acsz'
prob_para={'f1_score_average':'binary',#f値の計測タイプ指定
           #binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)#chatgptより
           'subnum':100,'sub_low':0.02,'sub_high':0.02,
           'dt_depth':3, 'depth_low':3, 'penalty_dnum':0.5,
           'penalty_ac':0.0, 'penalty_sz':1000, 'penalty_fm':0,'penalty_fl':1,
           'AC':1,'SZ':1,'FM':0,'FI':0,'feature_lock':None}
ga_para={'ngen':50, 'psize':100, 'pc':1, 'nvm':1, 'clones':False, 'vhigh':0.3}
genlist =list(range(0, ga_para['ngen']+1, int(ga_para['ngen']/5)))#グラフを書く世代(世代間)
genlist2=[ga_para['ngen']]
print(genlist,genlist2)
#%%
"""最適化問題を設定"""
"""ディレクトリを移動"""
os.mkdir(output_folder)#ディレクトリ作成
os.chdir(output_folder)#ディレクトリ移動
"""ハイパーパラメータの保存"""
df_hp = pd.DataFrame([prob_para,ga_para])
df_hp.to_csv('parameter.csv',index=False)
"""アルゴリズム実行(k分割交差検証)"""
ftime = open('time.txt', 'w', 1)#アルゴリズムの駆動時間を出力するファイルを作り､書き込む
for i, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    """ディレクトリ作成＆移動"""
    print('*** Run ', i, ' ***')
    run = 'run'+str(i)
    os.mkdir(run)
    os.chdir(run)# run i というファイルを作成しディレクトリを移動
    tic=timeit.default_timer()#tic-tocでアルゴリズムの駆動時間を計測
    """訓練＆テストデータ取得"""
    Xtr,Xte=X[train_idx],X[test_idx]#データをindexに応じて抽出
    ytr,yte=y[train_idx],y[test_idx]
    """データ前処理"""
    problem=Problem(Xtr,ytr,Xte,yte,xname,yname,f1_score_average=prob_para['f1_score_average'],
                subnum=prob_para['subnum'],sub_low=prob_para['sub_low'],sub_high=prob_para['sub_high'],
                dt_depth=prob_para['dt_depth'],depth_low=prob_para['depth_low'],penalty_dnum=prob_para['penalty_dnum'],
                penalty_ac=prob_para['penalty_ac'],penalty_fl=prob_para['penalty_fl'],penalty_sz=prob_para['penalty_sz'],penalty_fm=prob_para['penalty_fm'],
                AC=prob_para['AC'],SZ=prob_para['SZ'],FM=prob_para['FM'],FI=prob_para['FI'],feature_lock=prob_para['feature_lock'])
    problem.preprocessing(i)#データ前処理
    """GAフェーズ実行"""
    pop = nsgaii(evaluate = problem.fitness, 
           select = ea.binary_tournament_dom_cd, recombine = ea.crossover_1p, 
           mutate = ea.bit_flip_mutation, initype='binary', seed=i, 
           psize=ga_para['psize'], nobj=problem.nobj, nvar=problem.genelen, vlow=0, 
           vhigh=ga_para['vhigh'], ngen=ga_para['ngen'], pcx=ga_para['pc'], 
           pmut=ga_para['nvm']/problem.genelen, keepclones = ga_para['clones'])#進化計算フェーズ 
    """終了処理"""   
    toc=timeit.default_timer()#時間計測終了
    ftime.write('Nsgaii run' +str(i)+ ' ' + str(toc - tic) + ' seconds\n')#駆動時間の書き込み
    os.chdir('..')#ディレクトリを一つ前に戻す   
ftime.close()#駆動時間に関するファイルを閉じる
#%%
"""グラフ作成フェーズ※随時設定"""
"""ほしい散布図リストを設定"""
xl=['AC(mk)','AC(ev)','AC(test)','F1(mk)','F1(ev)','F1(test)','AC(ev)','F1(test)','F1(ev)']#x軸
yl=['size',  'size',  'size',    'size',  'size',  'size','AC(test)','importance_uniformity','importance_uniformity']#y軸
"""ディレクトリ作成"""
os.mkdir('graph')
os.mkdir('graph/gen_graph')
os.mkdir('graph/run_graph')
"""csvファイルのロード"""
pops_list=[]
for r in range(k):
    for g in genlist:
        pop_df=pd.read_csv('run{}/pop_g{}.csv'.format(str(r),str(g)))
        pop_df['gen']=g
        pop_df['run']=r
        pops_list.append(pop_df)
df_all=pd.concat(pops_list)
"""グラフ作成（世代間比較）"""
for r in range(k):
    for x, y in zip(xl, yl):
        pltdf=df_all[df_all['gen'].isin(genlist)&df_all['run'].isin([r])]#dfをあるrunのあるgensに限定
        fig = plt.figure(figsize = (8,5))
        ax=sns.scatterplot(data=pltdf, x=x, y=y, hue="gen",alpha=0.75,palette="Set1")
        fig.savefig('graph/gen_graph/{}_{}_scat_run{}'.format(str(x),str(y),str(r)))
        plt.close(fig)
"""グラフ作成(実行回間比較)"""
for g in genlist2:
    for x, y in zip(xl, yl):
        pltdf=df_all[df_all['run'].isin(list(range(10)))&df_all['gen'].isin([g])]#dfをあるrunのあるgensに限定
        fig = plt.figure(figsize = (8,5))
        ax=sns.scatterplot(data=pltdf, x=x, y=y, hue="run",alpha=0.75,palette="Set1")
        fig.savefig('graph/run_graph/{}_{}_scat_gen{}'.format(str(x),str(y),str(g)))
        plt.close(fig)
# %%
#front0のうちデプロイするモデルを定める
import pandas as pd
k=10
topn=20


top_list=[]
middle_list=[]
bottom_list=[]
for i in range(k):
    df=pd.read_csv('run'+str(i)+'/pop_g50.csv')
    #df=df[df['front'] == 0]
    df=df.sort_values(by='F1(ev)', ascending=False)
    middle_index = len(df) // 2 
    top_index = 1
    bottom_index = len(df) * 8//10
    middle_row = df.iloc[middle_index,:].to_dict()
    top_row = df.iloc[top_index,:].to_dict()
    bottom_row = df.iloc[bottom_index,:].to_dict()
    middle_list.append(middle_row)
    top_list.append(top_row)
    bottom_list.append(bottom_row)

dft=pd.DataFrame(top_list)
dft.to_csv('top.csv',index=False)
dfm=pd.DataFrame(middle_list)
dfm.to_csv('middle.csv',index=False)
dfb=pd.DataFrame(bottom_list)
dfb.to_csv('bottom.csv',index=False)

# %%
