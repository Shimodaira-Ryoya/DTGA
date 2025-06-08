#%%
"""ライブラリのインポート"""
import os
import numpy as np
import pandas as pd
import timeit
from sklearn.model_selection import train_test_split
from module.problem2 import Problem
from module.nsgaii import nsgaii
from module import ea_base as ea
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
"""データセットの準備※随時調整"""
df=pd.read_csv('../../database/redwine_dataset.csv')
X_df=df.iloc[:,1:-1]
y_df=df.iloc[:,-1]
X =X_df.values
xn=X_df.columns.tolist()
y,yn= pd.factorize(y_df)#値を整数値にエンコード
#y,yn=y_df.values,[0,1]
print('table_shape;',X_df.shape)
print('class sample;',y_df.value_counts())
# %%
"""パラメータ設定※随時調整"""
output_folder='../../output/method2/wine_red/ho0.67_dp20_alldata_fmsz'
k=3
prob_para={'f1_score_average':'weighted',#f値の計測タイプ指定#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)#chatgptより
           'all_data_evaluation':True,#評価データを全トレーニングデータにするか
           'subnum':100,'sub_low':0.02,'sub_high':0.02,
           'dt_depth':0, 'depth_low':20, 'penalty_dnum':[0.8,0.95],
           'penalty_ac':0, 'penalty_sz':1000, 'penalty_fm':0,'penalty_fl':1,
           'AC':0,'SZ':1,'FM':1,'FI':0,
           'feature_lock':None, 'dtinfo_store':False}
ga_para={'ngen':50, 'keepsimilar':False,'psize':100, 'pc':1, 'nvm':1, 'clones':False, 'vhigh':0.7}
genlist =list(range(0, ga_para['ngen']+1, int(ga_para['ngen']/5)))#グラフを書く世代(世代間)
genlist2=[ga_para['ngen']]
print(genlist,genlist2)
#%%
"""ディレクトリを移動"""
os.mkdir(output_folder)#ディレクトリ作成
os.chdir(output_folder)#ディレクトリ移動
"""ハイパーパラメータの保存"""
df_hp = pd.DataFrame([prob_para,ga_para])
df_hp.to_csv('parameter.csv',index=False)
"""アルゴリズム実行(holdout検証)"""
ftime = open('time.txt', 'w', 1)#アルゴリズムの駆動時間を出力するファイルを作り､書き込む
for i in range(k):
    """ディレクトリ作成＆移動"""
    print('*** Run ', i, ' ***')
    run = 'run'+str(i)
    os.mkdir(run)
    os.chdir(run)# run i というファイルを作成しディレクトリを移動
    tic=timeit.default_timer()#tic-tocでアルゴリズムの駆動時間を計測
    """訓練＆テストデータ取得"""
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.33, stratify=y, random_state=i)
    print('ytr class;',Counter(ytr))
    print('yte class;',Counter(yte))
    """データ前処理"""
    problem=Problem(Xtr,ytr,Xte,yte,xn,yn,f1_score_average=prob_para['f1_score_average'],all_data_evaluation=prob_para['all_data_evaluation'],
                subnum=prob_para['subnum'],sub_low=prob_para['sub_low'],sub_high=prob_para['sub_high'],
                dt_depth=prob_para['dt_depth'],depth_low=prob_para['depth_low'],penalty_dnum=prob_para['penalty_dnum'],
                penalty_ac=prob_para['penalty_ac'],penalty_fl=prob_para['penalty_fl'],penalty_sz=prob_para['penalty_sz'],penalty_fm=prob_para['penalty_fm'],
                AC=prob_para['AC'],SZ=prob_para['SZ'],FM=prob_para['FM'],FI=prob_para['FI'],feature_lock=prob_para['feature_lock'],dtinfo_store=prob_para['dtinfo_store'])
    problem.preprocessing(i)#データ前処理
    """GAフェーズ実行"""
    pop = nsgaii(evaluate = problem.fitness, 
           select = ea.binary_tournament_dom_cd, recombine = ea.crossover_1p, 
           mutate = ea.bit_flip_mutation, initype='binary', seed=i, 
           psize=ga_para['psize'], nobj=problem.nobj, nvar=problem.genelen, vlow=0, 
           vhigh=ga_para['vhigh'], ngen=ga_para['ngen'], pcx=ga_para['pc'], 
           pmut=ga_para['nvm']/problem.genelen, keepclones = ga_para['clones'],keepsimilar=ga_para['keepsimilar'])#進化計算フェーズ 
    """終了処理"""   
    toc=timeit.default_timer()#時間計測終了
    ftime.write('Nsgaii run' +str(i)+ ' ' + str(toc - tic) + ' seconds\n')#駆動時間の書き込み
    os.chdir('..')#ディレクトリを一つ前に戻す   
ftime.close()#駆動時間に関するファイルを閉じる
#%%
"""グラフ作成フェーズ※随時設定"""
"""ほしい散布図リストを設定"""
xl=['AC(ev)','AC(test)','F1(ev)','F1(test)','F1(ev)','AC(ev)','mk_ratio','mk_ratio']#x軸
yl=['size',  'size',    'size',    'size',  'F1(test)', 'AC(test)','F1(test)','AC(test)']#y軸
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
