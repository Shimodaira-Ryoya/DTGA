"""データ処理"""
import pandas as pd
from .dominate import get_non_dominated_solutions as dom
def deal(ga_datas,others,dominate):
    """standard_analysisのデータ処理
    args:ga_datas:list(dict)
        others:list(dict)
        dominate:dict
    """
    dfs=[]
    """ga_datasの処理"""
    if ga_datas!=0:
        for i in range(len(ga_datas)):
            pops=[]
            for run in ga_datas[i]['run']:
                for gen in ga_datas[i]['gen']:
                        pop_df=pd.read_csv(ga_datas[i]['folder']+'run{}/pop_g{}.csv'.format(str(run),str(gen)))
                        pop_df['gen']=gen
                        pop_df['run']=run
                        pops.append(pop_df)
            datai_df=pd.concat(pops)
            datai_df=dom(datai_df,dominate['x'],dominate['y'],dominate['x_order'],dominate['y_order'])
            datai_df['type']=ga_datas[i]['type']
            dfs.append(datai_df)
        
    """othersの処理"""
    if others!=0:
        for i in range(len(others)):
            df=pd.read_csv(others[i]['file'])
            df['type']=others[i]['type']
            df=dom(df,dominate['x'],dominate['y'],dominate['x_order'],dominate['y_order'])
            dfs.append(df)
        
    df_all=pd.concat(dfs)
    """dominate deal"""
    if dominate['USE']==True:
        df_all=df_all[df_all["dom"]=="nondom"]
        
    return df_all