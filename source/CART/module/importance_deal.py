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