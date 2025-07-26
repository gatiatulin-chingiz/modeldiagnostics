import pandas as pd
import numpy as np



def Gini(y_true, y_pred, weights=None, policy = 'accurate'):
    """ 
    Gini index. 
    Version = 3.0
    ----------------------------
    y_true - target values
    y_pred - estimated target values
    weights - sample weights. in y_true and y_pred - value per unit. 
                По-русски, при использовании весов y_true и y_pred - таргет и оценка на единицу.
                Например, для моделирования частоты: y_true - частота на год (на единицу), y_pred - её оценка. weights - экспозиция.
    policy - [default:'accurate','fast'], how to calculate: accurate - can be long, but accurate. 
                fast - can be inaccurate if pred containes duplicates.
    """
    
    def special_cumsum(v, w):
        return sum(np.cumsum([0]+list(v*w)[:-1])*w + v*w*(w+1)/2)
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    if n_samples == 0:
        return np.nan
    
    if weights is not None:
        weights = np.array(weights)
        assert y_true.shape == weights.shape
        assert all(weights >= 0)
        
        arr = np.array([y_true, y_pred, weights]).transpose()
        true_order, true_weights = arr[np.lexsort(arr[:, [1,0]].T, axis=-1)][::-1,[0,2]].T
        
        if policy == 'accurate':
            idxs = arr[:,1].argsort()[::-1]
            pred_order = pd.DataFrame(arr).groupby([1])[0].transform(np.mean)[idxs].values
            pred_weights = arr[:,2][idxs]
        elif policy == 'fast': 
            pred_order, pred_weights = arr[np.lexsort(arr[:, [0,1]].T, axis=-1)][::-1,[0,2]].T
        else:
            raise Exception('Incorrect policy.')
        L_true = special_cumsum(true_order,true_weights) / np.sum(true_order*true_weights)
        L_pred = special_cumsum(pred_order,pred_weights) / np.sum(pred_order*pred_weights)
        G_true = L_true - (np.sum(weights)+1)/2
        G_pred = L_pred - (np.sum(weights)+1)/2
    else:
        arr = np.array([y_true, y_pred]).transpose()
        true_order = arr[np.lexsort(arr[:, [1,0]].T, axis=-1)][::-1,0]
        if policy == 'accurate':
            pred_order = pd.DataFrame(arr).groupby([1])[0].transform(np.mean)[arr[:,1].argsort()].values[::-1]
        elif policy == 'fast': 
            pred_order = arr[np.lexsort(arr[:, [0,1]].T, axis=-1)][::-1,0]
        else:
            raise Exception('Incorrect policy.')
        L_true = np.cumsum(true_order) / np.sum(true_order)
        L_pred = np.cumsum(pred_order) / np.sum(pred_order)
        G_true = np.sum(L_true) - (n_samples+1)/2
        G_pred = np.sum(L_pred) - (n_samples+1)/2
        
    return G_pred/G_true