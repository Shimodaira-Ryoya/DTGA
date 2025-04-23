import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from . import dt_info 

class Problem:
    def __init__(self,Xtr,ytr,Xte,yte,xname,yname,
                 mkbit,evbit,plow_mk,phigh_mk,plow_ev,phigh_ev,ev_per,
                 dt_depth,depth_low,AC,SZ,FI,FM,f1_score_average,
                 penalty_ac=0,penalty_sz=5000,penalty_fm=0,penalty_fl=1):
        """dataset"""
        self.Xtr=Xtr#ndarray(np.float64)  最適化に用いる入力データセット
        self.ytr=ytr#ndarray(np.int32)    最適化に用いる出力データセット
        self.Xte=Xte#ndarray(np.float64)  精度評価に用いる入力データセット
        self.yte=yte##ndarray(np.int32)   精度評価に用いる出力データセット
        self.xn=xname#list                入力データセットの特徴量名リスト
        self.yn=yname#list                出力データセットのクラス名リスト
        """parameter"""
        self.mkbit=mkbit    #GAで決定木作成に用いるデータサブセットを何個作るか,mkbit=3なら2^3=8個作成、g2の長さ
        self.evbit=evbit    #GAで決定木評価に用いるデータサブセットを何個作るか,evbit=3なら2^3=8個作成、g3の長さ
        self.dt_depth=dt_depth#決定木の深度情報 g4の長さ
        self.depth_low=depth_low#CARTの最大深度の最低数　
        self.plow_mk =plow_mk  #決定木作成用データサブセットに決定木作成用データセットの何割のサンプルを引っ張ってくるか
        self.phigh_mk=phigh_mk #plow~phigh
        self.plow_ev =plow_ev  #決定木評価用データサブセットに決定木評価用データセットの何割のサンプルを引っ張ってくるか
        self.phigh_ev=phigh_ev #plow~phigh
        self.ev_per=ev_per     #学習データを決定木作成用データセットと決定木評価用データセットに分けるときの後者の割合
        self.f1_score_average=f1_score_average#f値の計測タイプ#binary(バイナリ),macro(均衡なデータ向き),weighted(不均衡なデータ向き),micro(全体の精度的なスコア)
        """fitness"""
        #遺伝子評価に用いる指標、0で不採用、1で採用
        self.AC=AC#精度：遺伝子から作成された決定木を決定木評価用データセットに通した時の正解率
        self.SZ=SZ#サイズ:作製された決定木のノード数
        self.FI=FI#特徴量重要度
        self.FM=FM#F-measure
        """penalty_border"""
        self.penalty_ac=penalty_ac#このボーダーを上回るor下回ると実行不可能解となる
        self.penalty_sz=penalty_sz#ペナルティ値の出し方に関しては仮、要検討
        self.penalty_fm=penalty_fm
        self.penalty_fl=penalty_fl
        """"""
        self.nobj=sum([self.AC,self.SZ,self.FI,self.FM])#評価関数の数
        self.xnum=self.Xtr.shape[1] #datasetの特徴量の個数
        self.genelen=self.xnum+self.mkbit+self.evbit+self.dt_depth#遺伝子の長さ
        
    def preprocessing(self,seed,stratify=True):
        """データセットの処理

        Args:
            seed (int): シード値
            stratify (bool):データ分割にて、サンプルのクラスの割合を同じにするかどうか. Defaults to True.
        """
        self.Xmk,self.Xev,self.ymk,self.yev=train_test_split(self.Xtr, self.ytr, 
                                                             test_size=self.ev_per, random_state=seed,
                                                             stratify=self.ytr if stratify else None)#stratifyがTrueならクラスのサンプル割合が均等になるよう分けられる
        self.submk_index_list=create_subset_index(2**self.mkbit,self.Xmk.shape[0],self.plow_mk,self.phigh_mk,seed)
        self.subev_index_list=create_subset_index(2**self.evbit,self.Xev.shape[0],self.plow_ev,self.phigh_ev,seed) 
        
    def fitness(self,gene):
        """入力ベクトル(gene)を基に決定木モデルを作成、評価
        Args:gene (list): 入力ベクトル バイナリ列
        """
        
        """遺伝子の読み込み"""
        gene1 = gene[ : self.xnum] if any(x != 0 for x in gene[:self.xnum]) else [1]*len(gene[:self.xnum])#もしgene1が全て0なら全て1に変える
        gene2 = gene[self.xnum : self.xnum + self.mkbit] if self.mkbit >= 1 else [0]#もしmkbitが0ならgene2=[0]
        gene3 = gene[self.xnum + self.mkbit : self.xnum + self.mkbit + self.evbit] if self.evbit >= 1 else [0] 
        gene4 = gene[self.xnum + self.mkbit + self.evbit : ] if self.dt_depth >=0 else [0]
        
        """遺伝子をデコード"""
        self.mk_i = convert_bit_dec(gene2)#採用するmk(i)のi
        self.ev_j = convert_bit_dec(gene3)#採用するev(j)のj
        depth = convert_bit_dec(gene4)     
        self.maxdepth=depth+self.depth_low #決定木の最大深度
        
        """使用するサンプルの洗い出し"""
        self.subXmk = self.Xmk[self.submk_index_list[self.mk_i],:]  
        self.subymk = self.ymk[self.submk_index_list[self.mk_i]]    
        self.subXev = self.Xev[self.subev_index_list[self.ev_j],:]  
        self.subyev = self.yev[self.subev_index_list[self.ev_j]]    
        
        """特徴量の抽出"""
        self.subXmk = delete_x(self.subXmk,gene1)#削減されたデータ
        self.subXev = delete_x(self.subXev,gene1)
        self.sxn = delete_xname(self.xn,gene1)#削減された特徴量名
        
        """モデル生成"""#決定木のパラメータに関して要検討
        self.clf=DecisionTreeClassifier(max_depth=self.maxdepth,class_weight='balanced',
                                        splitter='best',random_state=0)
        self.clf.fit(self.subXmk,self.subymk)#モデル学習
        
        """性能評価"""#どのデータで精度、F値を測るかは要検討
        self.dXte=delete_x(self.Xte,gene1)#Xteデータの特徴量削減データ
        self.size = self.clf.tree_.node_count#モデルのノード数
        self.ac   = accuracy_score(self.subymk,self.clf.predict(self.subXmk))#学習データでの精度
        self.ac2  = accuracy_score(self.subyev,self.clf.predict(self.subXev))#評価データでの精度
        self.ac3  = accuracy_score(self.yte,self.clf.predict(self.dXte))#がちテストデータでの精度
        self.fm  = f1_score(self.subymk,self.clf.predict(self.subXmk),average=self.f1_score_average)#学習データでのf値
        self.fm2 = f1_score(self.subyev,self.clf.predict(self.subXev),average=self.f1_score_average)#評価データでのf値
        self.fm3 = f1_score(self.yte,self.clf.predict(self.dXte),average=self.f1_score_average)#がちテストデータでのf値
        self.mk_ratio=self.subymk.shape[0]/self.ytr.shape[0]#学習サブセットのサンプル使用率(トレーニングデータ全体における割合)
        self.ev_ratio=self.subyev.shape[0]/self.ytr.shape[0]#評価サブセットのサンプル採用率(トレーニングデータ全体における割合)
        self.importance=importance_deal(gene1,self.clf.feature_importances_)#特徴量ごとの重要度を取得
        self.imp_uniformity=importance_uniformity(self.importance)#特徴量均一度の取得

        """決定木情報を持ってくる"""
        dtinfo=dt_info.DTinfo()
        dtinfo.read_clf(self.clf,self.sxn,self.yn)
        
        """評価値の決定"""
        fitness = []
        fitness +=[self.ac2]   if self.AC == 1 else []
        fitness +=[-self.size] if self.SZ == 1 else []
        fitness +=[self.fm2]   if self.FM == 1 else []
        fitness +=[self.imp_uniformity] if self.FI == 1 else []
        
        """ペナルティの計算"""#テスト未実施
        penalty=0
        penalty += (self.penalty_ac - self.ac2) if self.penalty_ac > self.ac2 else 0
        penalty += (self.penalty_fm - self.fm2) if self.penalty_fm > self.fm2 else 0
        penalty += (self.size - self.penalty_sz) if self.penalty_sz < self.size else 0
        penalty += (self.imp_uniformity - self.penalty_fl) if self.penalty_fl < self.imp_uniformity else 0 
        
        """データとして残す値の設定"""#テスト未実施
        #値にスカラー以外を入れる場合、別途ea_base.ind.detail_dealで処理を書き加える
        detail={'AC(mk)':round(self.ac,3), 'AC(ev)':round(self.ac2,3), 'AC(test)':round(self.ac3,3), 'size':self.size,
                'F1(mk)':round(self.fm,3), 'F1(ev)':round(self.fm2,3), 'F1(test)':round(self.fm3,3), 
                'mk(i)':self.mk_i, 'mk_ratio':round(self.mk_ratio,3), 'ev(j)':self.ev_j, 'ev_ratio':round(self.ev_ratio,3),
                #faverage
                'importance_uniformity':round(self.imp_uniformity,3),
                'importance_list':dict(zip(self.xn, self.importance))#ea_baseで特別な処理してる
                }
       
        return fitness,penalty,detail,dtinfo
    
    
def create_subset_index(n,d_size,plow,phigh,seed):
    """0~d_sizeまでの数をシャッフルし抽出してランダムに並べた配列をn個格納したリストを返す

        n (int): リストの個数
        d_size (int): 数の最大値
        plow (float): d_sizeに対して抽出する配列長さの最小割合（d_size=100,plow=0.1なら0~100の値が入った10の長さの配列が最小となる）
        phigh (float): d_sizeに対して抽出する長さの最大割合
        seed (int): シード値

    Returns:
        sub_index_list: list(ndarray)0~d_sizeの値がランダムに入った（重複なし）配列をn個持ったリスト
    """
    np.random.seed(seed)
    sub_index_list=[]
    for i in range(n):
        s=np.random.uniform(plow,phigh)#配列の抽出する割合を定める
        split_point=int(d_size*s)#配列を抽出する個数を定める
        index=np.random.permutation(range(d_size))#配列[0,1,2,3..]を作成し要素をランダムにシャッフル
        sub_index=np.sort(index[0:split_point])#配列を抽出し、昇順ソート
        sub_index_list.append(sub_index)
    return sub_index_list

def convert_bit_dec(bit_list):
    """bitのリストから変換した10進数を返す
    Args: bit_list (list,binary):0,1のリスト
    Returns: num:bit_listを10進数に直した数
    """
    num=0
    for i in range(len(bit_list)):
        num = num + 2**(len(bit_list)-i-1)*bit_list[i]
    return num

def delete_x(Xdata,xbit):
    """xbitにおいて0となる値があるインデックスと同じXdataのインデックスを削除
    Args:
        Xdata (ndarray): 特徴量X（入力ベクトル）のデータ、二次元配列
        xbit (_type_): 0or1のどのxiを採用するかのリスト、1で採用
    Returns:
        sXdata: 改新したデータ
    """
    notadx_index=[]
    for i in range(len(xbit)):
        if xbit[i]==0:
            notadx_index.append(i)
    sXdata=np.delete(Xdata,notadx_index,1)#Xdataは改変されない
    return sXdata 

def delete_xname(xnames,xbit):
    """listのxnamesから,xbitが0である要素と同じインデックスを削除する
    Args:
        xnames (list):
        xbit (list):0,1のリスト
    Returns:
        sxname: 改変後xnameリスト
    """
    sxname=[]
    for i in range(len(xbit)):
        if xbit[i]==1:
            sxname.append(xnames[i])
    return sxname

def importance_deal(gene1,valuelist):
    """特徴量重要度を特徴量ごとのリストとして保存

    Args:
        gene (list): 特徴量を採用するか1否か0のバイナリ列バイナリ列
        xname (list): 特徴量名リスト
        valuelist (ndarray): 特徴量重要度
    """
    imp_list=[]
    j=0
    for index in range(len(gene1)):
        if gene1[index]==1:
            imp_list.append(float(valuelist[j]))
            j+=1
        else:
            imp_list.append(0)
            
    return imp_list

def importance_uniformity(importance_list):
    """特徴量重要度の均一性を測る。式はジニ不純度の式と同様

    Args:
        importance_list (list,float): 特徴量重要度のリスト
    Return:
        uniformity(float):特徴量の均一性係数 0~1の範囲で0に近いほど不均一性(純度)が高い
    """
    uniformity=1
    for i in range(len(importance_list)):
        if importance_list[i]!=0:
            uniformity=uniformity-importance_list[i]**2
            
    return uniformity
    
if __name__ == '__main__':
    gene1=[0,0,0,0,0,1,0]
    valuelist=[]
    importance_deal(gene1,valuelist)
