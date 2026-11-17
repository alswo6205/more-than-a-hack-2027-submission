import os
import numpy as np
from paths import BASE
M=["J03WMX","J03WN1","J03WOH","J03WOY","J03WPY","J03WQQ","J03WR9"]
def runs(f):
    f=np.asarray(f); n=len(f)
    if n==0: return []
    ch=np.flatnonzero(np.r_[True,f[1:]!=f[:-1],True])
    return [(ch[i],ch[i+1]-1,f[ch[i]]) for i in range(len(ch)-1)]
def load(m):
    d=dict(np.load(os.path.join(BASE,f'{m}.npz'))); out=[]
    for p in (1,2):
        ix=np.flatnonzero(d['period']==p)
        e={k:d[k][ix] for k in ('xy','frame','gt','prob','bnd','pred','ball')}; e['teams']=d['teams']; e['m']=m; e['p']=p
        xy=e['xy']; t=e['teams']
        hx=np.nanmean(xy[:,t==-1,0]); ax=np.nanmean(xy[:,t==1,0])
        e['goal']={-1:(0. if hx<ax else 105.), 1:(105. if hx<ax else 0.)}   # own goal x per team
        gk={}
        for tm in (-1,1):
            g=e['goal'][tm]; dd=np.hypot(xy[:,t==tm,0]-g,xy[:,t==tm,1]-34)
            dd=np.where(np.isfinite(dd),dd,1e9); j=np.argmin(dd,1); idx=np.flatnonzero(t==tm)[j]
            gk[tm]=xy[np.arange(len(xy)),idx]            # [T,2] keeper = player nearest own goal
        e['gk']=gk
        out.append(e)
    return out
def inbox(x,y,g,depth=16.5,half=20.16):
    return (np.abs(x-g)<depth)&(np.abs(y-34)<half)
def gk_hold(e,dist=1.5,minlen=10):
    """truth: alive frames, ball within dist of defending keeper inside own box, >=2 s. returns list (s,e,team,medz)"""
    out=[]; b=e['ball']; alive=e['gt']!=2
    for tm in (-1,1):
        k=e['gk'][tm]; g=e['goal'][tm]
        f=alive&(np.hypot(b[:,0]-k[:,0],b[:,1]-k[:,1])<dist)&inbox(k[:,0],k[:,1],g)&(e['gt']==(0 if tm==-1 else 1))
        # close gaps <=2 frames
        r=runs(f)
        for i,(s,t_,v) in enumerate(r):
            if not v and 0<i<len(r)-1 and t_-s+1<=2: f[s:t_+1]=True
        for s,t_,v in runs(f):
            if v and t_-s+1>=minlen: out.append((s,t_,tm,float(np.median(b[s:t_+1,2]))))
    return sorted(out)
