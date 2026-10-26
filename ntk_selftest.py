import sys, json, time, numpy as np, torch, torch.nn.functional as F
import ntk_lib as L, ntk_pair as Pm
from model import LatentProbeModel
R='/vol/bitbucket/lbastian/latent-dynamics/runs_v2'
run=R+'/runs/mnist_probe_none_d16_s0_phase_n4000_wd0'
a=sys.argv[1].split(); opt_kind=a[0]; wd=float(a[1])
ctx=Pm.make_ctx(R+'/data',run+'/probe_index.npz',run+'/probe_train_index.npz',run+'/train_index.npz')
m=LatentProbeModel(16,latent_dim if False else 'mlp') if False else LatentProbeModel(16,'mlp')
m.load_state_dict(torch.load(run+'/model_final.pt',map_location='cpu')); m=m.to(L.DEV).train()
with torch.no_grad():
    for p in m.parameters(): p.add_(0.03*torch.randn_like(p)*p.std())
params=list(m.parameters())
opt=torch.optim.Adam(params,lr=1e-3,weight_decay=wd) if opt_kind=='adam' else torch.optim.SGD(params,lr=0.025,momentum=0.9,weight_decay=wd)
g=torch.Generator().manual_seed(1); pool=ctx['pool_idx']
def step(shuf=False):
    idx=pool[torch.randperm(len(pool),generator=g)[:128].numpy()]
    bi=torch.as_tensor(idx).to(L.DEV); x=ctx['Xfull'][bi]; y=ctx['lab_full'][bi]
    if shuf: y=y[torch.randperm(len(y),device=y.device)]
    opt.zero_grad(); F.cross_entropy(m(x)[0],y).backward(); opt.step(); return idx
hist=[]
for _ in range(40): hist.append(step())   # perturb final model w/ fresh optimizer state (non-trivial m, v)
def zp(): 
    with torch.no_grad(): return L.z_of([p for p in m.parameters()],ctx['Xp'])
za=zp(); sa=L.make_state(m,opt); i1=step(); hist.append(i1); zb=zp(); sb=L.make_state(m,opt); i2=step(); zc=zp(); sc=L.make_state(m,opt)
t=time.time()
out=Pm.analyze_pair(sb,sc,i2,ctx,obs_prev=zb-za,batch_prev_idx=i1,hist_batches=hist[::-1][:40],kernel=True)
print('time',time.time()-t, 'peakGB',torch.cuda.max_memory_allocated()/1e9)
print(json.dumps({k:v for k,v in out.items() if k not in('c_kernel',)},indent=1))
k=out['c_kernel']
for nm in k: print(nm,{a:(round(b,4) if isinstance(b,float) else '...') for a,b in k[nm].items()})
