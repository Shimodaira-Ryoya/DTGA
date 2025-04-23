import random, sys
import copy
import pandas as pd

class Individual(list):
    """一つの遺伝子"""
    def __init__(self, nobj, nvar, vlow, vhigh, initype):
        self.nvar = nvar#評価関数の個数
        self.nobj = nobj#遺伝子の変数の数
        self.fitness = ()#評価関数の値
        self.penalty=0 #ペナルティ0より大きい場合、実行不可能解に分類する
        self.rank = [-1.0] * 2#生存選択における個体のランクを示す
        self.dtinfo=None#決定木構造の情報
        self.detail=None#csvファイルに残すデータ
        if initype == 'binary':#遺伝子の変数タイプ binaryなら変数の範囲は0と1のみ
            for i in range(nvar):
                if random.random() < vhigh :#vhighは初期遺伝子生成においてある変数で1になる確率
                    self.append(1)
                else:
                    self.append(0)
        else:
            sys.exit("Init Error: init " + str(initype))

    def detail_deal(self):
        """csvファイルに残すデータを整理する
        具体的には各々データ情報+フロント&CD情報+特徴量重要度+遺伝子という順のリストを作成
        """
        """コンテナ値の分割&リスト化"""
        importance_dict=self.detail['importance_list']
        importance_columns = list(importance_dict.keys())
        importance_ratios  = list(importance_dict.values())
        """スカラー値をもつその他データを分割&リスト化"""
        s_detail = {k: v for k, v in self.detail.items() if k != 'importance_list'}#importancelistキーを除外した辞書を取得
        detail_columns = list(s_detail.keys())
        detail_values  = list(s_detail.values())
        """残す情報をリストにまとめる"""
        self.detail_columns =  detail_columns+['front','CD']
        self.detail_columns += ['feature_importance']+importance_columns
        self.detail_columns += ['gene']+['gene'+str(i) for i in range(len(self))]
        self.detail_values  =  detail_values+[round(x, 2) for x in self.rank]
        self.detail_values  += [None]+[round(x, 2) for x in importance_ratios]
        self.detail_values  += [None]+self       
        
class Population(list):
    """遺伝子集団（individualの集団）"""
    def __init__(self, size=0, nobj=0, nvar=0, vlow=0, vhigh=0, initype=0):
        """遺伝子individualをsize分生成"""
        for i in range(size):#sizeは遺伝子の数を示す。
            self.append(Individual(nobj, nvar, vlow, vhigh, initype))
    
    def printpop(self):
        """遺伝子それぞれのパラメータを出力"""
        for i in range(len(self)):
            print(i, self[i].printind()) 
            
    def pop_info_to_csv(self,directory):
        """遺伝子集団の情報をcsvファイルにまとめる
           directory(str):ファイルを保存するディレクトリ
        """
        df_values=[]
        for i in range(len(self)):
            self[i].detail_deal()
            df_values.append(self[i].detail_values)
        df_columns=self[0].detail_columns
        df =pd.DataFrame(df_values,columns=df_columns)
        df.to_csv(directory)
        
    def pop_dtinfo_to_csv(self,directory):
        """遺伝子集団の決定木の構造の情報をcsvファイルにまとめる"""
        for i in range(len(self)):
            self[i].dtinfo.write_csv(directory+'/clf'+str(i)+'.csv')


"""親選択、交叉、突然変異のメソッド"""

"""親選択"""
def binary_tournament(pop, n):
    """遺伝子集団のうち個体をランダムに二つ選び、性能を比較し(トーナメント)、優秀な方を親として返す
        fitnessで比較を行う
    Args:
        pop (gene.population): 遺伝子集団
        n (int): トーナメントを行う回
    Returns:
        parents: 親として返された個体リスト
    """
    parents = []#親集団リスト
    for k in range(0,n):#n回のトーナメント
        i = random.randint(0,len(pop)-1)#ランダムに個体を選択
        j = random.randint(1,len(pop)-1)
        j = (i+j)%len(pop)              #もうひとつ(被らないように) 個体選択
        if pop[i].fitness > pop[j].fitness:#性能比較
                                           #※fitness(tuple)比較は最初の要素の比較になると思われる(要検証)
            parents.append(copy.deepcopy(pop[i]))#copy.deepcopy：完全なる複製の生成、
        else:                                    #               コピー元の変更の影響を受けない
            parents.append(copy.deepcopy(pop[j]))
    return parents

def binary_tournament_dom_cd(pop, n):
    """遺伝子集団のうち個体をランダムに二つ選び、性能を比較し(トーナメント)、優秀な方を親として返す
       rankで比較を行う
    Args:
        pop (gene.population): 遺伝子集団
        n (int): トーナメントを行う回
    Returns:
        parents: 親として返された個体リスト
    """
    parents = []#親集団リスト
    for k in range(0,n):#n回のトーナメント
        i = random.randint(0,len(pop)-1)#ランダムに個体を選択
        j = random.randint(1,len(pop)-1)#もうひとつ(被らないように) 個体選択
        j = (i+j)%len(pop)
        if pop[i].rank[0] != pop[j].rank[0]:#性能比較、まずrank[0]同士の比較を行い、
                                            #これが同じならrank[1]同士の比較になる
            if pop[i].rank[0] < pop[j].rank[0]:
                parents.append(copy.deepcopy(pop[i]))#copy.deepcopy：完全なる複製の生成、コピー元の変更の影響を受けない                               #               コピー元の変更の影響を受けない
            else:
                parents.append(copy.deepcopy(pop[j]))
        else: 
            if pop[i].rank[1] > pop[j].rank[1]:
                parents.append(copy.deepcopy(pop[i]))
            else:
                parents.append(copy.deepcopy(pop[j]))       
    return parents

"""交叉"""
def crossover_1p(ind1, ind2):
    """一点交叉：交差点をポイントに遺伝子を切り取りこれを入れ替える
     Args:
        ind1,ind2:親からコピーされた子供、まだただの親のクローンの個体
    Returns:
        ind1,ind2:一点交叉を行った子供個体
        nvar_change:遺伝子が何個変更されたか
    """
    k = random.randint(0,len(ind1)-1)#交叉点をランダムに選択
    xind = copy.deepcopy(ind1)
    nvar_change = 0#遺伝子が何個変更されたか
    for i in range(k,len(ind1)):
        if ind1[i] != ind2[i]:
            nvar_change += 1
        ind1[i] = ind2[i]#ind1の交差点から最後までの遺伝子をind2の変数に書き換え
        ind2[i] = xind[i]#ind2の交差点から最後までの遺伝子をind1の変数に書き換え
    return nvar_change

"""突然変異"""
def bit_flip_mutation(ind, mutp):
    """突然変異:確率で遺伝子の変数をビット反転
    Args:
        ind (gene.individual): 遺伝子個体
        mutp (float): 突然変異率(遺伝子の変数一つに対して)
    Returns:
        ind:突然変異後の遺伝子個体
        nvar_change:遺伝子が何個変更されたか
    """
    nvar_change = 0#遺伝子が何個変更されたか
    for i in range(0,len(ind)):
        if random.random() < mutp:#突然変異率に応じてビット反転
            ind[i] = (ind[i]+1)%2
            nvar_change += 1
    return nvar_change