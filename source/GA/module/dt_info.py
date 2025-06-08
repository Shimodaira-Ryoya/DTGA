#%%
import numpy as np
import pandas as pd
from graphviz import Digraph

class DTinfo:
    """決定木の情報をノードi(i=0,1,2...n_nodes)ごとに格納
    """  
    def __init__(self):
        
        self.n_nodes=0 #(int):全ノード数
        self.f_average=-10000 #(float)特徴量数平均（葉ノードまでの特徴量数の平均) -10000=None
        self.feature=None #(ndarray):i番目のノードの分割に用いられる特徴量は何番目の特徴か※なければ-2
        self.threshold=None#(ndarray):i番目のノードの分割における閾値は何か※なければ-2？
        self.children_left=None #(ndarray):i番目のノードの子ノード(左)は何番目のノードか※なければ-1
        self.children_right=None#(ndarray):i番目のノードの子ノード(右)は何番目のノードか※なければ-1
        self.mother_node=None   #(ndarray):i番目のノードの親ノードは何番目のノードか※親がなければ-1
        self.values=None        #(ndarray):i番目のノードに分割されてきたデータのクラスごとの数
        self.node_depth=None    #(ndarray):i番目のノードの深さ
        self.is_leaves=None     #(ndarray(TorF))i番目のノードは葉ノードか
        self.xn=None#(list)データの特徴量名※特徴量削減で決定木を作製した場合は対応する特徴量名のみを残しほかを消すこと
        self.yn=None#(list)データのクラス名
        
    def read_clf(self,clf,xn,yn):
        """fit済みclfから決定木情報を抽出

        Args:
            clf (desiciontreeclassfier): fit済みclf
            xn(list):clfの作成に用いた特徴量名リスト
            yn(list):clfの作成に用いたクラス名リスト
        """
        self.xn=xn
        self.yn=yn
        self.n_nodes = clf.tree_.node_count
        self.feature = clf.tree_.feature
        self.threshold = clf.tree_.threshold
        self.children_left = clf.tree_.children_left
        self.children_right = clf.tree_.children_right
        self.values = np.squeeze(clf.tree_.value)
        self.node_depth = np.zeros(shape=self.n_nodes, dtype=np.int64)
        self.is_leaves = np.zeros(shape=self.n_nodes, dtype=bool)
        stack = [(0, 0)]#なにこれ
        while len(stack) > 0:
            node_id, depth = stack.pop()
            self.node_depth[node_id] = depth
            is_split_node = self.children_left[node_id] != self.children_right[node_id]
            if is_split_node:
                stack.append((self.children_left[node_id], depth + 1))
                stack.append((self.children_right[node_id], depth + 1))
            else:
                self.is_leaves[node_id] = True
    
        self.mother_node = np.zeros(shape=self.n_nodes, dtype=np.int64)#親ノード情報
        self.mother_node[0]=-1#根ノードの親ノードはなし
        for i in range(self.n_nodes):#i番目のノードの子ノードに親が誰か記録させる
            if self.children_left[i]!=-1:
                self.mother_node[self.children_left[i]]=i
            if self.children_right[i]!=-1:
                self.mother_node[self.children_right[i]]=i
    
        self.f_average=self.calculate_usef_ave()#平均使用特徴量数
                  
    def write_csv(self,directory):
        """csv形式で決定木情報をファイルに書き込む
        Args:
        directory(str):書き込むファイルのディレクトリ
        """  
        df = pd.DataFrame({
            'node_number': np.arange(self.n_nodes),#連番配列をつくる
            'feature': replace_number_toname(self.feature,self.xn),
            'threshold': self.threshold,
            'children_left': self.children_left,
            'children_right': self.children_right,
            'mother_node':self.mother_node,
            'node_depth':self.node_depth,
            'is_leaves':self.is_leaves
            })
        df_values=pd.DataFrame(self.values,columns=self.yn)#クラスごとのサンプル割合を示すdf#おそらくvaluesに全てのクラスが含まれていないとバグる
        df_values.insert(loc=0, column='values', value=np.nan)#dfの前にvalue列(空値)を追加
        df=pd.concat([df,df_values],axis=1)
        df.to_csv(directory)

    def read_csv(self,directory):
        """決定木情報をcsvファイルから読み込む
        Args:
            directory (str): 読み込むファイルのディレクトリ
        """
        df = pd.read_csv(directory)
        df_values=df.loc[:,'values':]
        dict = {col:df[col].values for col in df.columns}
        self.n_nodes = len(dict['node_number'])
        self.feature,self.xn = replace_name_tonumber(dict['feature'])
        self.threshold = dict['threshold']
        self.children_left = dict['children_left']
        self.children_right= dict['children_right']
        self.mother_node = dict['mother_node']
        self.node_depth = dict['node_depth']
        self.is_leaves = dict['is_leaves']
        self.values = df_values.loc[:,0:].iloc[:,1:].values
        self.yn = df_values.loc[:,0:].iloc[:,1:].columns.to_list()
        self.f_average=self.calculate_usef_ave()
        
    def script_info(self,columns=True):
        """決定木情報を出力：
        順に1.ノード番号 2.ノードの深さ 3.分割に採用する特徴量 4.分割の閾値
        5.左子ノードのノード番号 6.右子ノードのノード番号 7.ノードにおけるクラスごとのデータ量
        
        Args:
        columns(bool):情報タイトルを出すか否か
        """
        if columns:#情報タイトルを出すか否か
            print("{:<7} {:<7} {:<20} {:<7} {:<7} {:<7} {:<7} {:<7}".format("node_i","depth","feature","threhld","c_left","c_right","mother","value"))
        
        for i in range(self.n_nodes):
            if self.is_leaves[i]:
                print("{:<7} {:<7} {:<20} {:<7} {:<7} {:<7} {:<7} {:<7}"
                      .format(i,self.node_depth[i],"leaf","leaf","none","none",self.mother_node[i],str(self.values[i])))
            else:
                print("{:<7} {:<7} {:<20} {:<7} {:<7} {:<7} {:<7} {:<7}"
                      .format(i,self.node_depth[i],self.xn[self.feature[i]],str("{:.4g}".format(self.threshold[i])),
                              self.children_left[i],self.children_right[i],self.mother_node[i],str(self.values[i])))

    def plot_DT(self,title,pas="../output/Nsgaii/DTplot",type="pdf",view=False):
        """決定木の可視化

        Args:
            title (str): 決定木グラフのタイトル
            pas (str, optional): 決定木グラフを保存するフォルダまでのパス. Defaults to "../output/Nsgaii/DTplot".
            type (str, optional):ファイルの保存形式. Defaults to "pdf".
            view (bool, optional): グラフをすぐ見るか. Defaults to False.
        """
        
        # 有向グラフのインスタンス化
        g = Digraph()
        # 属性の指定
        g.attr('node', shape='square')
        for i in range(self.n_nodes):
            if self.is_leaves[i]==False:
                mother= 'node='+str(i)+'\n'+str(self.xn[self.feature[i]])+"<="+str(self.threshold[i])+'\n'+str(self.values[i])
                if self.yn==None:
                    leaf_left  = 'node='+str(self.children_left[i])+'\n' +'class='+str(np.argmax(self.values[self.children_left[i]]))+'\n'+str(self.values[self.children_left[i]])
                    leaf_right = 'node='+str(self.children_right[i])+'\n'+'class='+str(np.argmax(self.values[self.children_right[i]]))+'\n'+str(self.values[self.children_right[i]])
                else:
                    leaf_left  = 'node='+str(self.children_left[i])+'\n' +'class='+str(self.yn[np.argmax(self.values[self.children_left[i]])])+'\n'+str(self.values[self.children_left[i]])
                    leaf_right = 'node='+str(self.children_right[i])+'\n'+'class='+str(self.yn[np.argmax(self.values[self.children_right[i]])])+'\n'+str(self.values[self.children_right[i]])
                left  = 'node='+str(self.children_left[i])+'\n'+str(self.xn[self.feature[self.children_left[i]]])+"<="+str(self.threshold[self.children_left[i]])+'\n'+str(self.values[self.children_left[i]])
                right = 'node='+str(self.children_right[i])+'\n'+str(self.xn[self.feature[self.children_right[i]]])+"<="+str(self.threshold[self.children_right[i]])+'\n'+str(self.values[self.children_right[i]])
                if self.is_leaves[self.children_left[i]]==False:
                    g.edge(mother,left)
                else:
                    g.edge(mother,leaf_left)
                if self.is_leaves[self.children_right[i]]==False:
                    g.edge(mother,right)
                else:
                    g.edge(mother,leaf_right)
        g.render(pas+"/"+title, format=type, view=view)


    def predict(self,testx,output=True):
        """DT情報を基にテストデータに対するクラスを予測、予測に用いる条件を表示
        Args:
            testx (ndarray): テストデータ
        Returns:
            pred:テストデータに対する予測クラス
        """
        #testxを葉ノードまで分類しその分類過程を記録する
        node=0
        nodelist=[0]#testxが通ったノードの番号
        directlist=[]#testxの分割に際し、左(閾値以下)に通るか右(閾値以上)に通るか、左:-1 右:1
        while self.feature[node]!=-2:#葉ノードかどうかの判定
            if testx[self.feature[node]] <= self.threshold[node]:
                #nodeで指定される特徴量においてtestxの値が閾値を下回れば左へ下がる
                #print("{:<15} <= {:<10}".
                #      format(self.xn[self.feature[node]],self.threshold[node]))
                node=self.children_left[node]#左子ノードに移動
                nodelist.append(node)#ノード情報の記録
                directlist.append(-1)#通る道が左か右かの記録
            else:
                #print("{:<15} >  {:<10}".
                #      format(self.xn[self.feature[node]],self.threshold[node]))
                node=self.children_right[node]
                nodelist.append(node)
                directlist.append(1)
        
        #nodelist,directlistを基に使用する特徴量、閾値の整理
        featurelist=[]#使用する特徴量、重複はなし
        morelist=[]#testxを上回った閾値 上の特徴量リストに要素が対応
        lesslist=[]#testxを下回った閾値
        # ex)   featurelist  1  2  5  6  8
        #       morelist     10 ×　×　3　7
        #       lesslist     3　6　5　×　2
        for i in range(len(nodelist)-1):
            if self.feature[nodelist[i]] not in featurelist:
                #使う条件の特徴量が重複してなかったら新たに書き込み
                featurelist.append(self.feature[nodelist[i]])
                if directlist[i]==-1:
                    lesslist.append(None)
                    morelist.append(self.threshold[nodelist[i]])
                else:
                    lesslist.append(self.threshold[nodelist[i]])
                    morelist.append(None)
            else:#重複且つ閾値の範囲がより近づいたら上書き
                same=find_elements(featurelist,self.feature[nodelist[i]])
                if directlist[i]==-1: 
                    if morelist[same] is None or self.threshold[nodelist[i]]<morelist[same]:
                        morelist[same]=self.threshold[nodelist[i]]
                else:
                    if lesslist[same] is None or self.threshold[nodelist[i]]>lesslist[same]:
                        lesslist[same]=self.threshold[nodelist[i]]
        
        #表示
        if output==True:       
            for i in range(len(featurelist)):
                if morelist[i] is None:
                    print("{:<10} < {:<15}    {:<10}".
                        format(lesslist[i],self.xn[featurelist[i]]," "))
                elif lesslist[i] is None:
                    print("{:<10}   {:<15} <= {:<10}".
                        format(" ",self.xn[featurelist[i]],morelist[i]))
                else:
                    print("{:<10} < {:<15} <= {:<10}".
                        format(lesslist[i],self.xn[featurelist[i]],morelist[i]))
                
        
        pred=np.argmax(self.values[node])
        
        if output==True:
            if self.yn is None:
                print("class {}".format(pred))
            else:
                print("class {}".format(self.yn[pred]))
        
        return pred
                      

    def track_from_i_node(self,i):
        """指定ノードから根ノードまで追跡し各ノードで使われる特徴量を取得
        ※重複は省略

        Args:
            i (int): 追跡を始めるノード番号

        Returns:
            use_feature(set):使われる特徴量番号
        """
        use_feature=set()
        while self.mother_node[i]!=-1:
            i=self.mother_node[i]
            use_feature.add(self.feature[i])
        return use_feature
    
    def calculate_usef_ave(self,output=False):
        """決定木のある予測に対し使われる特徴量数の平均を求める

        Returns:
            self.avef: 特徴量数平均（葉ノードまでの特徴量数の平均）
        """
        n_feature_leaf_to_root=[]
        for i in range(self.n_nodes):
            if self.is_leaves[i]==True:
                f=self.track_from_i_node(i)
                n_feature_leaf_to_root.append(len(f))
        self.avef=sum(n_feature_leaf_to_root)/len(n_feature_leaf_to_root)
        if output==True:
            print("use_feature_num_average={}".format(self.avef))
        return self.avef
    
    def calculate_usef_all(self,output=False):
        """決定木全体で使われる特徴量とその数を返す

        Args:
            output (bool, optional): 結果を出力するか否か. Defaults to False.
        """
        feature=set(self.feature)
        feature.remove(np.int64(-2))
        print(feature)
        
def replace_number_toname(number,name):
    """決定木情報整理用、番号の配列から、番号に対応する名前が入った配列をつくる
        番号が-2の場合'LEAF NODE'と返す
    Args:
        number (ndarray): 番号の配列（番号の範囲は0~len(name)及び-2
        name (list): 名前のリスト、各番号が名前に対応する
    Return:
        name_array:番号配列が名前配列に置き換わった
    """
    name_list=[]
    for i in range(len(number)):
        if number[i]==-2:
            name_list.append('LEAF_NODE')
        else:
            name_list.append(name[number[i]])
    name_array=np.array(name_list)
    return name_array

def replace_name_tonumber(name_array):
    """決定木情報整理用、名前の配列を番号の配列number_listに変換し、それに対応する名前のリストxnを作成
        値が'LEAF NODE'の場合-2を返す
    Args:
        name_array (array): 名前配列
    Return:
        number(array),name(list):番号配列、名前一覧リスト
    """
    name=[]
    number=[]
    for i in range(len(name_array)):
        if name_array[i]=='LEAF_NODE':
            number.append(-2)
        else:
            if name_array[i] not in name:
                name.append(name_array[i])
            number.append(name.index(name_array[i]))
    return number,name
                
def find_elements(list, value):
    """valueと同じ値を持つlistの要素番号を返す(一番最初の要素のみ)"""
    for i in range(len(list)):
        if value==list[i]:
            return i
    
    
if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_digits
    from dt_info import DTinfo
    import numpy as np
    import pandas as pd

    digits = load_digits()
    X = digits.data       # shape=(1797, 64)
    y = digits.target     # shape=(1797,)
    clf=DecisionTreeClassifier()
    clf.fit(X,y)
    feature_names = [f'pixel_{i}' for i in range(digits.data.shape[1])]
    class_names = digits.target_names
    dtinfo=DTinfo()
    dtinfo.read_clf(clf,feature_names,class_names)
    df=dtinfo.write_csv("unl")
    df.head()
    