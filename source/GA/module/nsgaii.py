import random, copy, os
from . import ea_base as ea
from . import moea_base as moea

def nsgaii_survival_selection(pop, popsize):
    """fitnessから非支配レベルfrontと混雑距離cdを計算しソーティング、生存選択を行う
    Args:
        pop (population): 遺伝子集団
        popsize (int): 生き残らせる個体数
    """
    
    nobj = pop[0].nobj#評価関数の数
    
    """実行不可能解(penalty>0)の処理"""
    pop_inf=ea.Population()#空のpopを生成（実行不可能解集団）
    size=len(pop)
    for i in range(size-1, -1, -1):#size-1,size-2...2,1,0と連番
        if pop[i].penalty>0:#i番目の個体が実行不可能解なら
            pop_inf.insert(0, pop[i])#実行不可能解集団に解を挿入
            pop.remove(pop[i])#元の集団から解は消え去る
    
    if len(pop)!=0:#実行可能解が存在すれば、それを非支配レベルで分ける
        fronts = moea.non_dominated_sorting(pop)#fronts:list　0番目にはflont0の個体集団、1番目にはflont1の個体集団...が入ってる
    else:
        fronts=ea.Population()#なければ空のpopを作成
    #この時、元のpopは空
    
    #popsizeのキャパを超えるまでfront0の個体集団,front1の個体集団...という順に空のpopにfront(x)の個体集団まるまる詰め込む
    i = 0
    if len(fronts)!=0:
        while len(pop) < popsize and len(pop) + len(fronts[0]) <= popsize:
            fi = fronts.pop(0)#frontsの0番目の要素を返し、さらにこれをリストから消去
            moea.front_rank(fi, i)#fi(同フロント集団)の個体のindividual.rank[0]にフロントランクを書き込む
            moea.crowding_distance(fi, nobj)#fiの個体のindividual.rank[1]に混雑距離を書き込む
            pop.extend(fi)#pop(list)の末尾にfi(list)をくっつける
            i += 1#フロントが一つ下がることを意味する
            if len(fronts)==0:
                break
        
    
    #popsizeのキャパを超えたら同一フロント内で混雑距離でソーティングし足きりを行う
    if len(pop) < popsize and len(fronts)!=0:
        fi = fronts.pop(0)
        moea.front_rank(fi, i)
        moea.crowding_distance(fi, nobj)
        fi.sort(key = lambda ind: ind.rank[1], reverse = True)
        pop.extend(fi[: popsize - len(pop)])
    
    #実行不可能解の処理、フロントを同一で１００００とする、またCDを書き込むところをペナルティを負に変換した値を書き込む
    if len(pop) < popsize:
        moea.front_rank(pop_inf,10000)
        for ind in pop_inf:
            ind.rank[1]=-ind.penalty
        pop_inf.sort(key = lambda ind: ind.rank[1],reverse = True)
        pop.extend(pop_inf[: popsize - len(pop)])

def nsgaii(evaluate=None, select=None, recombine=None, mutate=None, 
    seed=None, psize=None, nobj=None, nvar=None, vlow=None, vhigh=None, 
    initype=None, ngen=None, pcx=None, pmut=None, keepclones = False):
    """NSGA-llのアルゴリズムに基づいた処理を行う
    Args:
        evaluate(function):遺伝子評価の関数 
        select(function):親選択の関数
        recombine(function):交叉の関数
        mutate(function):突然変異の関数
        seed(int):遺伝子集団のランダムな初期集団生成時(及び交叉確率)のシード値
        psize(int):遺伝子集団の個体の数
        nvar(int):遺伝子の変数の数
        vlow,vhigh(float):遺伝子の変数の範囲(vlow~vhigh)、initypeがbinaryならvhighは初期個体生成において一つの変数が1になる確率
        initype(str):変数のタイプ
        ngen(int):世代数
        pcx(float):交叉確率
        pmut(float):突然変異確率(一つの変数に対して)
        keepclones(bool):親と全く同じ遺伝子を残すか否か
    """
    random.seed(seed)
    """ Initial population  """#初期集団生成
    pop = ea.Population(size=psize, nobj=nobj, nvar=nvar, vlow=vlow, 
                        vhigh=vhigh, initype=initype)
    
    """ Evaluate the initial population """
    for ind in pop:
        fitness,penalty,detail,dtinfo = evaluate(ind)
        ind.fitness=tuple(fitness)
        ind.penalty=penalty
        ind.detail=detail
        ind.dtinfo=dtinfo

    """ Output the population """
    nsgaii_survival_selection(pop, psize)#評価によるソーティング(非支配レベルと混雑距離）+下位個体を淘汰

    """store population informations"""
    pop.pop_info_to_csv('pop_g0.csv')#遺伝子情報（表現型情報等）を保存
    os.mkdir("DTinfo_gen0")#決定木構造情報を保存する
    pop.pop_dtinfo_to_csv('DTinfo_gen0')
    print(' --Generation 0')
   
    for g in range(1, ngen+1):
        """ Select the next generation individuals """#親選択　
        parents = select(pop, psize)#ea_base参照(親選択法)
        """ Clone the selected individuals """#親のクローンを生成
        offspring = copy.deepcopy(parents)

        """ Apply crossover and mutation on the offspring """#クローン個体を二つ選んで交叉 突然変異
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < pcx:
                recombine(ind1, ind2)#ea_base参照(交叉法)
            mutate(ind1, pmut)#ea_base参照(突然変異法)
            mutate(ind2, pmut)

        """ Evaluate offspring population """#子集団(クローン個体)を評価
        for ind in offspring:
            fitness,penalty,detail,dtinfo = evaluate(ind)
            ind.fitness=tuple(fitness)
            ind.penalty=penalty
            ind.detail=detail
            ind.dtinfo=dtinfo
            
        """ Delete clones """#親と全く同じ遺伝子は抹殺
        if keepclones == False:
            join_pop = []
            for ind in (pop+offspring):
                if ind not in join_pop:
                    join_pop.append(ind)
            pop[:] = join_pop[:]
        else:
            pop.extend(offspring)

        nsgaii_survival_selection(pop, psize)#評価ソート＆淘汰
      
        """ Output the population """
        if g%(ngen/10) == 0:         
            pop.pop_info_to_csv('pop_g{}.csv'.format(str(g)))#遺伝子情報（表現型情報等）を保存
            os.mkdir("DTinfo_gen{}".format(str(g)))#決定木構造情報を保存する
            pop.pop_dtinfo_to_csv('DTinfo_gen{}'.format(str(g)))
            print(' --Generation ', g)

    print('Ends Nsgaii')
    return pop