import pandas as pd
def dominates(find1, find2,xorder,yorder):
    """find1(ベクトル)がfind2(ベクトル)を支配しているか測定する
       find1がfind2を支配するとは、
       意味はfind1がfind2の完全上位互換であることを指し、
       定義はベクトルのすべての要素でfind1>=find2であり、いずれかの要素でfind1>find2であることを指す
    Args:
        find1 (list or tuple): 支配する側のベクトル
        find2 (list or tuple): 支配される側のベクトル
    Returns:
        dom: find1はfind2を支配しているかtrue/false
    """
    dom = False         #find1はfind2を支配しているかのフラグ
    better = 0          #ある要素で、find1>find2が成立したらカウントされる
    better_or_equal = 0 #ある要素で、find1>=find2が成立したらカウントされる
    nobj = len(find1)   #ベクトルの次元
    
    #for i in range(0,nobj):#要素ごとにfind1>find2とfind1>=find2の関係の成立の有無をカウント
    if xorder==1:
        if (find1[0] >= find2[0]):
            better_or_equal += 1
            if find1[0] > find2[0]:
                better += 1
    else: 
        if (find1[0] <= find2[0]):
            better_or_equal += 1
            if find1[0] < find2[0]:
                better += 1
    if yorder==1:
        if (find1[1] >= find2[1]):
            better_or_equal += 1
            if find1[1] > find2[1]:
                better += 1
    else: 
        if (find1[1] <= find2[1]):
            better_or_equal += 1
            if find1[1] < find2[1]:
                better += 1
    
    if (better_or_equal == nobj) and better >= 1:
        dom = True#すべての要素でfind1>=find2であり、いずれかの要素でfind1>find2なら支配関係は成立
    return dom

def get_non_dominated_solutions(df,x,y,xorder,yorder):
    """dfで指定された2つのカラム(x,y)を見て、
    データポイントそれぞれが他のデータ全てに対しフロント0か否かを判定しその情報を付加して返す
    Args:
        df (pddataframe):pandasのデータフレーム
        x(str):dfにあるカラム名1
        y(str):dfにあるカラム名2
        xorder(int):xが最大化か最小化か、xorder=1で最大化、それ以外で最小化
        yorder(int):yが最大化か最小化か、yorder=1で最大化、それ以外で最小化
    Returns:
        df(pddataframe):argのdfに"dom"というカラムが追加され
        それぞれのデータポイントにおいてフロント０のデータは"nondom"、被支配解のデータは"dom"となっている
    """
    size = len(df)#遺伝子集団の個体数
    dom_count = [0] * size

    #j番目の個体がほかのいくつの個体に支配されているかカウント
    #dom_countが0の個体は非支配解であることを意味する
    xtdf=df.loc[:,[x,y]]
    for j in range(size):
        for i in range(size):
            if i != j:
                if dominates(xtdf.iloc[i].values.tolist(), xtdf.iloc[j].values.tolist(),xorder,yorder):
                    dom_count[j] += 1
                    break
                    
                 
    df["dom"] = "dom"
    dom=df.loc[:,"dom"]
    for i in range(size):
        if dom_count[i] == 0:
            dom.iloc[i]="nondom"

    return df
