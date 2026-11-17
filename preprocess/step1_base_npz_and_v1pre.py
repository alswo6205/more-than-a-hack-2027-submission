"""Per-match base npz (positions, ball, original labels) + probabilities of the original V1 model (v1pre_prob).
  python preprocess/step1_base_npz_and_v1pre.py [match ...]
inputs : BENCH_INPUTS/<m>_input.csv (players), work/ball5/<m>_ball5.csv (ball, status, possession)
output : work/base/<m>.npz
"""
import sys, os
import numpy as np, pandas as pd, torch
from paths import THIRD_PARTY, BENCH_INPUTS, BALL5, BASE, V1_PRE, MATCHES
sys.path.insert(0, THIRD_PARTY)
from pa_sm.pipeline import load_checkpoint, predict_arrays
from pa_sm.decoders import semi_markov
M = MATCHES
torch.set_num_threads(2)
os.makedirs(BASE, exist_ok=True)
model,cfg,prior,dev=load_checkpoint(V1_PRE,'cpu')
for m in (sys.argv[1:] or M):
    inp=pd.read_csv(os.path.join(BENCH_INPUTS,f'{m}_input.csv')); b=pd.read_csv(os.path.join(BALL5,f'{m}_ball5.csv'))
    assert (inp.frame_id.values==b.frame_id.values).all()
    hs=[c[:-2] for c in inp.columns if c.startswith('home_') and c.endswith('_x')]; as_=[c[:-2] for c in inp.columns if c.startswith('away_') and c.endswith('_x')]
    sl=hs+as_; teams=np.zeros(46,np.int8); teams[:len(hs)]=-1; teams[len(hs):len(sl)]=1
    xy=np.full((len(inp),46,2),np.nan,np.float32)
    xy[:,:len(sl),0]=inp[[s+'_x' for s in sl]].values; xy[:,:len(sl),1]=inp[[s+'_y' for s in sl]].values
    gt=np.where(b.status.values==0,2,np.where(b.poss.values==1,0,1))
    prob=np.zeros((len(inp),3)); bnd=np.zeros((len(inp),2)); lab=np.zeros(len(inp),np.int8)
    for p in (1,2):
        ix=np.flatnonzero(inp.period.values==p)
        s,bd=predict_arrays(model,cfg,xy[ix],teams,dev)
        prob[ix]=s;bnd[ix]=bd;lab[ix]=semi_markov(s,bd,cfg['decoder'],prior,5)
    np.savez_compressed(os.path.join(BASE,f'{m}.npz'),xy=xy,teams=teams,frame=inp.frame_id.values,period=inp.period.values,gt=gt,prob=prob,bnd=bnd,pred=lab,
                        ball=np.c_[b.x.values+52.5,34-b.y.values,b.z.values])
    print(m,'acc',(lab==gt).mean(),'AD acc',((lab==2)==(gt==2)).mean(),flush=True)
