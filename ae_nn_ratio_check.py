import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[k]="8"
import sys, json, numpy as np
sys.path.insert(0,"/homes/lbastian/latent-dynamics")
import ae_analysis as A, ae_memorization as M
DS=sys.argv[1]
AN="/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/"+{"mnist":"","fashion_mnist":"fashion","cifar10":"cifar10"}[DS]
KIND,D,NF=("conv",64,50000) if DS=="cifar10" else ("mlp",16,60000)   # nonlinear-AE kind, latent dim, full-N size
TMP=os.environ.get("NN_TMP","/tmp/claude-2292/s"); os.makedirs(TMP,exist_ok=True)
RUNS="/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/runs/"; V2="/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/"
rid=lambda N,wd,s: f"{DS}_ae_{KIND}_d{D}_s{s}_n{N}_wd{wd}"
rd=lambda r: RUNS+r if os.path.isdir(RUNS+r) else V2+r
Xtr_all,_=A.images(DS,True); Xte_all,_=A.images(DS,False)
def pcaref(r,N):
    d_=rd(r); pte=np.load(d_+"/probe_index.npz")["indices"]; ptr=np.load(d_+"/probe_train_index.npz")["indices"]
    pool=np.load(d_+"/train_index.npz")["indices"] if N!="full" else np.arange(len(Xtr_all))
    Xp=Xtr_all[pool].astype(np.float64); mu=Xp.mean(0)
    if N=="full": w,v=np.linalg.eigh(np.cov(Xp.T)); P=v[:,::-1][:,:D]
    else: P=np.linalg.svd(Xp-mu,full_matrices=False)[2][:D].T
    ztr=(Xtr_all[ptr]-mu)@P; zte=(Xte_all[pte]-mu)@P
    return M.nn_med(zte,ztr)/M.nn_med(ztr,ztr,True)
rows=[]
for N in (250,1000,4000,"full"):
  for wd in ("0","1e-4"):
    for s in range(3):
      r=rid(N,wd,s); J=json.load(open(f"{AN}/runs/{r}.json")); m=np.load(f"{AN}/memorization/{r}.npz"); R=np.load(f"{AN}/runs/{r}.npz")
      rows.append(dict(N=N,wd=float(wd),s=s,logN=np.log(NF if N=="full" else N),y=np.log(J["final_recon_test"]/J["final_recon_train"]),gap=J["final_recon_gap"],nn=float(m["nn_ratio"][-1]),pca=pcaref(r,N),r=r,m=m,R=R,cfg=(N,wd)))
  print(DS,N,"done",flush=True)
import pickle; pickle.dump([{k:v for k,v in x.items() if k not in("m","R")} for x in rows],open(f"{TMP}/rows_{DS}.pkl","wb"))
def res(X,v): v=np.array(v,float); return v-X@np.linalg.lstsq(X,v,rcond=None)[0]
def pc(sub,key,ykey="y",extra=()):
    X=np.c_[np.ones(len(sub)),[x["logN"] for x in sub],[x["wd"]>0 for x in sub]].astype(float)
    return np.corrcoef(res(X,[x[key] for x in sub]),res(X,[x[ykey] for x in sub]))[0,1]
out=[]
P=lambda *a: (print(*a),out.append(" ".join(str(x) for x in a)))
P(f"## {DS}")
# 1 reproduce
sub=[x for x in rows if x["N"]!="full"]
P("1. partial(nn,y|logN,wd): all24",round(pc(rows,"nn"),3),"N<=4000",round(pc(sub,"nn"),3),"| N>=1000 (excl 250)",round(pc([x for x in rows if x["N"]!=250],"nn"),3),"| excl250&full",round(pc([x for x in rows if x["N"] in(1000,4000)],"nn"),3))
P("   raw spearman(nn,y) all24",round(A.spearman(np.array([x["nn"] for x in rows]),np.array([x["y"] for x in rows])),3))
# per-config means
for N in (250,1000,4000,"full"):
  for wd in ("0","1e-4"):
    c=[x for x in rows if x["N"]==N and x["wd"]==float(wd)]
    P(f"   N={N} wd={wd}: nn {np.mean([x['nn'] for x in c]):.4f}±{np.std([x['nn'] for x in c]):.4f} pca {np.mean([x['pca'] for x in c]):.4f}±{np.std([x['pca'] for x in c]):.4f} y {np.mean([x['y'] for x in c]):.3f}±{np.std([x['y'] for x in c]):.3f}")
# 2a within config: demean within cfg, pooled correlation and per-cfg
def dm(key): 
    v=np.array([x[key] for x in rows],float)
    for cf in set(x["cfg"] for x in rows):
        ii=[i for i,x in enumerate(rows) if x["cfg"]==cf]; v[ii]-=v[ii].mean()
    return v
for lab_,keep in (("all",lambda x:True),("excl N=250",lambda x:x["N"]!=250),("N<=4000",lambda x:x["N"]!="full")):
    ii=[i for i,x in enumerate(rows) if keep(x)]
    a,b=dm("nn")[ii],dm("y")[ii]; n=len(ii)
    # permutation p within cfg
    rs=np.random.RandomState(0); obs=np.corrcoef(a,b)[0,1]; cnt=0
    cf=[rows[i]["cfg"] for i in ii]; groups=[[j for j,c in enumerate(cf) if c==u] for u in set(cf)]
    for _ in range(5000):
        bb=b.copy()
        for g in groups: bb[g]=b[rs.permutation(g)]
        cnt+=abs(np.corrcoef(a,bb)[0,1])>=abs(obs)
    P(f"2a. within-config pooled r(nn,y) [{lab_}] n={n}: {obs:.3f}  perm p={(cnt+1)/5001:.3f}")
    ap=np.corrcoef(dm("pca")[ii],b)[0,1]; P(f"    within-config r(pca_ref,y): {ap:.3f}; r(nn,pca)={np.corrcoef(a,dm('pca')[ii])[0,1]:.3f}")
# per-config r
pr=[]
for cf in sorted(set(x["cfg"] for x in rows),key=str):
    c=[x for x in rows if x["cfg"]==cf]; pr.append((cf,np.corrcoef([x["nn"] for x in c],[x["y"] for x in c])[0,1]))
P("   per-config r (n=3 each):"," ".join(f"{c[0]}/{c[1]}:{r:+.2f}" for c,r in pr))
# 2b leave one N out
for N in (250,1000,4000,"full"):
    s2=[x for x in rows if x["N"]!=N]
    P(f"2b. leave N={N} out: partial(nn,y|logN,wd) = {pc(s2,'nn'):.3f} (n={len(s2)})")
# 2e PCA ref
P("2e. partial(pca_ref,y) all24",round(pc(rows,"pca"),3),"N<=4000",round(pc(sub,"pca"),3))
for x in rows: x["exc"]=x["nn"]-x["pca"]
P("    partial(nn-pca,y) all24",round(pc(rows,"exc"),3),"N<=4000",round(pc(sub,"exc"),3),"excl250",round(pc([x for x in rows if x["N"]!=250],"exc"),3))
# partial of nn controlling also pca
def pc2(sub,key,ctrl):
    X=np.c_[np.ones(len(sub)),[x["logN"] for x in sub],[x["wd"]>0 for x in sub],[x[ctrl] for x in sub]].astype(float)
    return np.corrcoef(res(X,[x[key] for x in sub]),res(X,[x["y"] for x in sub]))[0,1]
P("    partial(nn,y|logN,wd,pca) all24",round(pc2(rows,"nn","pca"),3),"N<=4000",round(pc2(sub,"nn","pca"),3),"excl250",round(pc2([x for x in rows if x["N"]!=250],"nn","pca"),3))
# also log-N^2 flexible (N dummies)
def pcN(sub,key):
    Ns=sorted(set(str(x["N"]) for x in sub)); X=np.c_[[[1.0]+[float(str(x["N"])==n) for n in Ns[1:]]+[x["wd"]>0] for x in sub]].astype(float)
    return np.corrcoef(res(X,[x[key] for x in sub]),res(X,[x["y"] for x in sub]))[0,1]
P("   partial with N-dummies+wd (flexible N): nn all24",round(pcN(rows,"nn"),3),"pca",round(pcN(rows,"pca"),3),"nn-pca",round(pcN(rows,"exc"),3))
# 2d time-resolved
res_t=[];res_d=[];res_pl=[]
for x in rows:
    if x["N"]=="full": pass
    m,R=x["m"],x["R"]
    st=m["steps"]; assert np.array_equal(st,R["steps"]),x["r"]
    g=np.log(R["sc_recon_test"]/R["sc_recon_train"]); nn=m["nn_ratio"]; ok=np.isfinite(g)&np.isfinite(nn)&(st>0)
    g,nn,ls=g[ok],nn[ok],np.log(st[ok])
    from numpy import corrcoef as cc
    d=cc(np.diff(g),np.diff(nn))[0,1]
    X=np.c_[np.ones(len(ls)),ls,ls**2]; pt=cc(res(X,g),res(X,nn))[0,1]
    res_t.append((x["N"],x["wd"],A.spearman(g,nn),d,pt,len(g)))
P("2d. time-resolved per run: spearman(level) | corr first-diffs | partial on poly2(log step); mean by config")
for N in (250,1000,4000,"full"):
  for wd in (0.0,1e-4):
    c=np.array([r[2:5] for r in res_t if r[0]==N and r[1]==wd],float)
    P(f"   N={N} wd={wd}: rho {c[:,0].mean():+.2f} dcorr {c[:,1].mean():+.2f} partial_t {c[:,2].mean():+.2f}  (n_snap={[r[5] for r in res_t if r[0]==N][0]})")
c=np.array([r[2:5] for r in res_t if r[0]!="full"],float); P("   mean N<=4000:",np.round(c.mean(0),3),"frac runs partial_t>0:",(c[:,2]>0).mean(),"frac dcorr>0:",(c[:,1]>0).mean())
# time-resolved nn vs pca? pca ref timeless. Also nn trajectory summary
for N in (250,1000):
    m=[x["m"]["nn_ratio"] for x in rows if x["N"]==N and x["wd"]==0.0 and x["s"]==0][0]; st=[x["m"]["steps"] for x in rows if x["N"]==N and x["s"]==0][0]
    P(f"   nn_ratio traj N={N} s0 (steps {st[0]}..{st[-1]}): "+" ".join(f"{v:.2f}" for v in m[::max(1,len(m)//12)]))
open(f"{TMP}/out_{DS}.txt","w").write("\n".join(out))
