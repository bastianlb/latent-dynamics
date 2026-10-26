import os
os.environ.setdefault("OMP_NUM_THREADS","2")
import glob, json, re, sys, collections
import numpy as np, csv
R='/vol/bitbucket/lbastian/latent-dynamics/runs'
OUT='/vol/bitbucket/lbastian/latent-dynamics/dynamics'
os.makedirs(OUT,exist_ok=True)
rows=[]
for d in sorted(glob.glob(R+'/*/')):
    d=d.rstrip('/'); rid=os.path.basename(d)
    if not os.path.exists(d+'/config.json') or rid.startswith('_'): continue
    c=json.load(open(d+'/config.json'))
    fs=sorted(glob.glob(d+'/latents/step_*.npz'))
    steps=[];eps=[]
    for f in fs:
        with np.load(f) as z: steps.append(int(z['step'])); eps.append(float(z['epoch']))
    with np.load(fs[-1]) as z: acc=float(z['test_acc']); N=z['latents'].shape[0]
    steps=np.array(steps); eps=np.array(eps)
    sched=c.get('log_schedule','linear')
    if c.get('dense_early',0): sched+='+dense_early'
    rows.append(dict(run_id=rid,dataset=c['dataset'],objective=c.get('objective'),latent_norm=c.get('latent_norm'),
      latent_dim=c['latent_dim'],seed=c['seed'],encoder_arch=c.get('encoder_arch'),lr=c['lr'],batch_size=c['batch_size'],
      epochs=c['epochs'],total_steps=int(steps.max()),n_snapshots=len(steps),
      n_snap_first_epoch=int(((eps<=1.0)).sum()),n_snap_first100=int(((steps>0)&(steps<=100)).sum()),
      n_snap_first10=int(((steps>0)&(steps<=10)).sum()),
      min_nonzero_step=int(steps[steps>0].min()),log_schedule=sched,n_probe=N,final_test_acc=acc,
      has_latents_train=os.path.isdir(d+'/latents_train'),snapshot_steps=' '.join(map(str,steps))))

cols=list(rows[0].keys())
with open(OUT+'/inventory.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
def vc(k): return dict(sorted(collections.Counter(r[k] for r in rows).items(),key=lambda x:str(x[0])))
fixed=[];varied=[]
for k in ['dataset','objective','latent_norm','latent_dim','seed','encoder_arch','lr','batch_size','epochs','n_probe','log_schedule','total_steps','n_snapshots']:
    v=vc(k); (varied if len(v)>1 else fixed).append((k,v))
def md(header,data):
    return ['| '+' | '.join(header)+' |','|'+'---|'*len(header)]+['| '+' | '.join(str(x) for x in r)+' |' for r in data]
L=['# Inventory and data-adequacy summary','',f'{len(rows)} runs (all with `latents/`; `latents_train/` present in {sum(r["has_latents_train"] for r in rows)}).','',
'## Control parameters that vary','']+[f'- **{k}**: {v}' for k,v in varied]+['','## Held fixed','']+[f'- **{k}**: {v}' for k,v in fixed]
L+=['','## Not config keys / never varied','- optimizer type, weight decay, train-set size (full train set), label noise (none), LR schedule, augmentation.']
by=collections.defaultdict(list)
for r in rows: by[r['dataset']].append(r)
data=[]
for ds,rs in by.items():
    data.append([ds,len(rs),rs[0]['total_steps'],min(r['n_snapshots'] for r in rs),max(r['n_snapshots'] for r in rs),np.median([r['n_snap_first_epoch'] for r in rs]),np.median([r['n_snap_first100'] for r in rs]),max(r['n_snap_first10'] for r in rs),min(r['min_nonzero_step'] for r in rs),'%.3f-%.3f'%(min(r['final_test_acc'] for r in rs),max(r['final_test_acc'] for r in rs))])
L+=['','## Snapshot density','']+md(['dataset','runs','total_steps','snaps_min','snaps_max','med snaps<=epoch1','med snaps in (0,100]','max snaps in (0,10]','first nonzero step','final test acc'],data)
L+=['','Schedule is linear (stride from config log_every=50, plus step 0) in every run; no log or dense_early. Steps 1-49 are entirely unobserved, and only steps 50 and 100 fall in the first 100 steps.','']
cf=collections.defaultdict(list)
for r in rows: cf[(r['dataset'],r['objective'],r['latent_norm'],r['latent_dim'])].append(r)
L+=['## Configs','']+md(['dataset','objective','latent_norm','latent_dim','n_seeds','mean final acc'],[list(k)+[len({r['seed'] for r in v}),'%.3f'%np.mean([r['final_test_acc'] for r in v])] for k,v in cf.items()])
open(OUT+'/inventory_summary.md','w').write('\n'.join(L))
for r in rows: print({k:v for k,v in r.items() if k!='snapshot_steps'})
