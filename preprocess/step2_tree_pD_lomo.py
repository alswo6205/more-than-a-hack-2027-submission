"""Leave-one-match-out tree D probability (tree_pD) on Sportec with the deadball-style features (5 Hz).
For each match, a HistGradientBoosting model trained on the other 6 matches predicts it.
  python preprocess/step2_tree_pD_lomo.py      -> work/tree_prob.npz  (keys <match>_<period>)
(The evaluation table that followed in the original script is not needed for the dataset and was removed.)
"""
import numpy as np,warnings;warnings.filterwarnings('ignore')
import features as FT
from common import M,load
from paths import TREE
from sklearn.ensemble import HistGradientBoostingClassifier
def tree_feats(e):
    X=e['xy'].astype(float).copy();X[...,0]-=52.5;X[...,1]-=34;team=(e['teams']==1).astype(int)
    keep=e['teams']!=0;X=X[:,keep];team=team[keep]
    v=np.full(X.shape[:2],np.nan);d=np.hypot(*np.moveaxis(np.diff(X,axis=0),-1,0))*5;d[d>12]=np.nan;v[1:]=d
    a=np.full_like(v,np.nan);a[1:]=np.abs(np.diff(v,axis=0))*5
    F0,n0=FT.frame_features(X[...,0],X[...,1],v,a,team);f0=FT._fill(F0);fc,_=FT.context_features(f0,n0)
    return np.hstack([f0,fc]).astype(np.float32)
E=[e for m in M for e in load(m)]
for e in E:e['F']=tree_feats(e)
print('features',E[0]['F'].shape,flush=True)
for test in M:
    X=np.vstack([e['F'] for e in E if e['m']!=test]);y=np.concatenate([e['gt']==2 for e in E if e['m']!=test])
    clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.08,max_leaf_nodes=31,random_state=0).fit(X,y)
    for e in E:
        if e['m']==test:e['pt']=clf.predict_proba(e['F'])[:,1]
    print(test,'tree trained',flush=True)
np.savez_compressed(TREE,**{f"{e['m']}_{e['p']}":e['pt'] for e in E})
