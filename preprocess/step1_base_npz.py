"""Per-match base npz (positions, ball, original labels).
  python preprocess/step1_base_npz.py [match ...]
inputs : BENCH_INPUTS/<m>_input.csv (players), work/ball5/<m>_ball5.csv (ball, status, possession)
output : work/base/<m>.npz
"""
import sys, os
import numpy as np, pandas as pd
from paths import BENCH_INPUTS, BALL5, BASE, MATCHES
M = MATCHES
os.makedirs(BASE, exist_ok=True)
for m in (sys.argv[1:] or M):
    inp=pd.read_csv(os.path.join(BENCH_INPUTS,f'{m}_input.csv')); b=pd.read_csv(os.path.join(BALL5,f'{m}_ball5.csv'))
    assert (inp.frame_id.values==b.frame_id.values).all()
    hs=[c[:-2] for c in inp.columns if c.startswith('home_') and c.endswith('_x')]; as_=[c[:-2] for c in inp.columns if c.startswith('away_') and c.endswith('_x')]
    sl=hs+as_; teams=np.zeros(46,np.int8); teams[:len(hs)]=-1; teams[len(hs):len(sl)]=1
    xy=np.full((len(inp),46,2),np.nan,np.float32)
    xy[:,:len(sl),0]=inp[[s+'_x' for s in sl]].values; xy[:,:len(sl),1]=inp[[s+'_y' for s in sl]].values
    gt=np.where(b.status.values==0,2,np.where(b.poss.values==1,0,1))
    np.savez_compressed(os.path.join(BASE,f'{m}.npz'),xy=xy,teams=teams,frame=inp.frame_id.values,period=inp.period.values,gt=gt,
                        ball=np.c_[b.x.values+52.5,34-b.y.values,b.z.values])
    print(m,len(inp),'frames',flush=True)
