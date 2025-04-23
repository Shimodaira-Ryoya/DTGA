from . import ea_base as ea

def dominates(find1, find2):
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
    
    for i in range(0,nobj):#要素ごとにfind1>find2とfind1>=find2の関係の成立の有無をカウント
        if (find1[i] >= find2[i]):
            better_or_equal += 1
            if find1[i] > find2[i]:
                better += 1
    
    if (better_or_equal == nobj) and better >= 1:
        dom = True#すべての要素でfind1>=find2であり、いずれかの要素でfind1>find2なら支配関係は成立
    return dom

def get_non_dominated_solutions(pop):
    """遺伝子集団を非支配解集団とそれ以外の集団に分ける
    Args:
        pop (population):遺伝子集団
    Returns:
        ndpop: 非支配解集団
    """
    size = len(pop)#遺伝子集団の個体数
    dom_count = [0] * size

    #j番目の個体がほかのいくつの個体に支配されているかカウント
    #dom_countが0の個体は非支配解であることを意味する
    for i in range(0,size):
        for j in range(0,size):
            if i != j:
                if dominates(pop[i].fitness, pop[j].fitness):
                    dom_count[j] += 1

    ndpop = ea.Population()#個体数0のpopulationを作成、非支配解集団
    for i in range(size-1, -1, -1):
        if dom_count[i] == 0:
            ndpop.insert(0, pop[i])#非支配解集団に非支配解を挿入
            pop.remove(pop[i])#元の集団から被支配解は消え去る
    return ndpop

def non_dominated_sorting(pop):
    """解のフロント分けを行う、若いindexからflont0の個体集団、flont1の個体集団...となる
    Args:
        pop (population):遺伝子集団
    Returns:
        fronts: フロントごとの遺伝子集団のリスト
    """
    fronts = []
    while (len(pop) > 0):
        fi  = get_non_dominated_solutions(pop)#popから非支配解集団fiを抽出し返す
        fronts.append(fi)
    return fronts

def front_rank(front, drank, i_index=0):
    """同フロントの個体集団frontの個体一つ一つのrankにdrank(フロントがいくつか)を書き込む
    Args:
        front (population): 同フロントの個体集団
        drank (int): 個体のindividual.rankに書き込む数字、フロントランクをあらわす
        i_index (int, optional): 個体のindividual.rank(list)のどの要素に書き込むか. Defaults to 0.
    """
    for ind in front:
        ind.rank[i_index] = drank
     
def crowding_distance(front, nobj,i_index=1):
    """front(同一フロントの個体集団)で個体間の混雑距離を求め、個体の.rankに記入
    Args:
        front (population): 同一フロントの遺伝子集団
        nobj (int): 評価関数の数
        i_index (int, optional):個体individual.rank(list)のどの要素に混雑距離を書き込むか Defaults to 1.
    """
    nind = len(front)#個体集団の個数
    INFINITY = 1e+15#無限大
    EPS = 1e-10#最小
    #個体すべての混雑距離cd(=individual.rank[i_index])を初期化
    for i in range(0, nind):
        front[i].rank[i_index] = 0.0

    for i in range (0, nobj):
        #lambda:一行で関数定義する ind(引数):ind.fitness[i](返り値)
        #frontの要素それぞれをindとしてind.fitness[i]を参照し降順でソート
        front.sort(key = lambda ind: ind.fitness[i], reverse = True)
        #fitnessの一番大きい値と一番小さい値を覚える
        maxf = front[0].fitness[i]
        minf = front[nind-1].fitness[i]
        #一番大きい値と一番小さい値をもつ要素の混雑距離は無限大とする
        front[0].rank[i_index] += INFINITY
        front[nind-1].rank[i_index] += INFINITY
        #ほかの要素の混雑距離を計算、
        # 一方の隣の個体の値からもう一方の隣の個体の値の差を求め、その次元における評価値の最大値と最小値の
        #範囲から標準化を行い、これを１次元における距離とする
        #これを全次元分行い、これを単純に加算したものを混雑距離とする
        for j in range(1, nind-1):
            d = front[j-1].fitness[i] - front[j+1].fitness[i]
            front[j].rank[i_index] += abs(d)/abs(EPS + maxf - minf)