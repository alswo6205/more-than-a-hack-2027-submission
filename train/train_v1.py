"""V1 (player-only attention + GRU, third_party/pa_sm) trained on the middle label (gt_mid).
Train on 6 Sportec matches, early stopping / model selection on J03WPY. Same as train_and_eval_combo.py with lab='mid'.
  python train/train_v1.py [--epochs N]      -> work/train_out/v1_mid.pt  (= models/v1_mid.zip)
"""
import argparse, copy, shutil, time, numpy as np
from common_train import D, M, TRAIN, VAL, LAB, OUT, dev
from pa_sm.pipeline import fit
from pa_sm.data import save_match
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); a = ap.parse_args(); T0 = time.time()
V1CFG={'seed':42,'preprocessing':{'fps':5,'source_fps':25,'pitch_x':105.0,'pitch_y':68.0,'max_slots':46,'max_speed_mps':12.0,'pressure_radius_m':5.0},
 'augmentation':{'flip_x':True,'flip_y':True,'shuffle_tokens':True,'position_noise_m':0.0,'dropout_probability':0.0,'duplicate_probability':0.0},
 'model':{'feature_dim':11,'embed_dim':64,'heads':4,'attention_layers':2,'seeds':4,'frame_dim':128,'hidden_size':64,'gru_layers':2,'dropout':0.1},
 'training':{'window':100,'train_stride':50,'inference_stride':50,'batch_size':8,'epochs':30,'lr':0.0003,'weight_decay':0.0001,'boundary_weight':0.2,'max_boundary_pos_weight':20.0,'early_stop_patience':7,'num_workers':2 if dev=='cuda' else 0,'grad_clip':1.0},
 'decoder':{'max_segment_seconds':180,'switch_cost_ha':2.0,'switch_cost_ad':2.5,'boundary_weight':0.4,'duration_weight':0.15,'duration_clip':5.0,'viterbi_switch_cost_ha':2.0,'viterbi_switch_cost_ad':2.5}}   # = V1 repo configs/base.yaml (num_workers changed for speed only)
V1CFG['training']['epochs'] = a.epochs
def v1_files(lab):
    d=f'{OUT}/v1data_{lab}';paths={}
    for m in M:
        z=D[m];paths[m]=[]
        for p in (1,2):
            ix=np.flatnonzero(z['period']==p);f=f'{d}/{m}_p{p}.npz'
            save_match(f,z['xy'][ix],z['teams'],z['frame'][ix],z['gt_'+lab][ix].astype(np.int64),z['period'][ix]);paths[m].append(f)
    return paths
paths=v1_files(LAB);print(f'==== V1 training, label {LAB} ====',flush=True)
best=fit(copy.deepcopy(V1CFG),[f for m in TRAIN for f in paths[m]],paths[VAL],f'v1_{LAB}','auto')
shutil.copy(best,f'{OUT}/v1_{LAB}.pt');shutil.rmtree(f'{OUT}/v1data_{LAB}',ignore_errors=True)
print(f'saved {OUT}/v1_{LAB}.pt  {(time.time()-T0)/60:.1f} min')
