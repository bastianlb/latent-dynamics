import os
from latentdyn._threads import cap_threads
cap_threads(4, force=True)
import sys, json, glob, re, numpy as np
from latentdyn.ae import analysis as A
DS=sys.argv[1] if len(sys.argv)>1 else "fashion_mnist"
AN="/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/"+{"mnist":"","fashion_mnist":"fashion","cifar10":"cifar10"}[DS]
# per-dataset conventions: nonlinear-AE kind, main latent dim, comparison dims (d-sweep), full-N size
KIND,D0,DSW,NF=("conv",64,(16,),50000) if DS=="cifar10" else ("mlp",16,(8,64),60000)
RUNS="/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/runs/"; V2="/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/"
def ms(x): x=np.array(x,float); return f"{np.nanmean(x):.4g}±{np.nanstd(x):.2g}"
def rid(kind,d,N,wd,s): return f"{DS}_ae_{kind}_d{d}_s{s}_n{N}_wd{wd}"
def load(r):
    J=json.load(open(f"{AN}/runs/{r}.json")); R=np.load(f"{AN}/runs/{r}.npz"); return J,R
def rd(r): return RUNS+r if os.path.isdir(RUNS+r) else V2+r
cfgs=[(KIND,D0,N,wd) for N in (250,1000,4000,"full") for wd in ("0","1e-4")]+[("linear",D0,N,"0") for N in (250,1000,4000,"full")]+[(KIND,d,N,"0") for d in DSW for N in (1000,"full")]
print("== 1. generalization (te, tr, abs gap, rel gap%, onset epoch(rel>10%), test-min epoch(step), final/min-1 %)")
for kind,d,N,wd in cfgs:
    out=[]
    for s in range(3):
        J,R=load(rid(kind,d,N,wd,s)); st=R["steps"]; tr=R["sc_recon_train"]; te=R["sc_recon_test"]
        Np=NF if N=="full" else N; ep=st*128/Np
        rel=te/tr-1; i=np.where(rel>0.10)[0]; on=ep[i[0]] if len(i) else np.nan
        im=int(np.argmin(te)); out.append((te[-1],tr[-1],te[-1]-tr[-1],100*rel[-1],on,ep[im],st[im],100*(te[-1]/te[im]-1)))
    o=np.array(out); print(f"{kind} d{d} N={N} wd={wd}: "+" | ".join(ms(o[:,j]) for j in range(8)))
# PCA reference
print("== 3. pixel-PCA reference at d16 (probe images on top-16 PCs of N-pool): spread_lab, var ratio; vs AE/linear final spread_lab/var")
Xtr_all,_=A.images(DS,True); Xte_all,_=A.images(DS,False)
from latentdyn.ae import memorization as M
for N in (250,1000,4000,"full"):
    rr=[]
    for s in range(3):
        d_=rd(rid(KIND,D0,N,"0",s)); pte=np.load(d_+"/probe_index.npz")["indices"]; ptr=np.load(d_+"/probe_train_index.npz")["indices"]
        pool=np.load(d_+"/train_index.npz")["indices"] if N!="full" else np.arange(len(Xtr_all))
        Xp=Xtr_all[pool].astype(np.float64); mu=Xp.mean(0); U,S,Vt=np.linalg.svd(Xp-mu,full_matrices=False) if N!="full" else (None,None,None)
        if N=="full":
            C=np.cov(Xp.T); w,v=np.linalg.eigh(C); Vt=v[:,::-1][:,:D0].T
        P=Vt[:D0].T
        ytr=np.load(d_+"/probe_train_index.npz")["indices"]; _,yt=A.images(DS,True); _,yte_all=A.images(DS,False)
        ztr=(Xtr_all[ptr]-mu)@P; zte=(Xte_all[pte]-mu)@P; ltr=yt[ptr]; lte=yte_all[pte]
        rs=np.random.RandomState(0); sl=[];vr=[]
        for _ in range(5):
            idx=[]
            for c in range(10):
                ic=np.where(lte==c)[0]; k=int(round(len(ptr)*(ltr==c).mean())); idx+=list(rs.choice(ic,min(k,len(ic)),replace=False))
            idx=np.array(idx); sl.append(M.pooled_within(ztr,ltr)/M.pooled_within(zte[idx],lte[idx])); vr.append(M.totvar(ztr)/M.totvar(zte[idx]))
        rr.append((np.mean(sl),np.mean(vr)))
    rr=np.array(rr)
    mem=json.load(open(AN+"/memorization/results.json"))
    f=lambda kind,wd: [mem[rid(kind,D0,N,wd,s)]["final"] for s in range(3)]
    ae=f(KIND,"0"); li=f("linear","0")
    print(f"N={N}: PCA ref spread_lab {ms(rr[:,0])} var {ms(rr[:,1])} | MLP spread_lab {ms([a['spread_lab'] for a in ae])} var {ms([a['var_ratio'] for a in ae])} nn_ratio {ms([a['nn_ratio'] for a in ae])} dgm {ms([a['dgm_ratio'] for a in ae])} white {ms([a['var_ratio_white'] for a in ae])}| linear spread_lab {ms([a['spread_lab'] for a in li])} var {ms([a['var_ratio'] for a in li])} nn {ms([a['nn_ratio'] for a in li])}")
print("== classifier (matched phase runs) spread_lab, var, nn_ratio, dgm")
for N in (250,1000,4000,"full"):
  for wd in ("0","1e-4"):
    k=[f"{DS}_probe_none_d16_s{s}_phase_n{N}_wd{wd}" for s in range(3)]
    try: x=[mem[r]["final"] for r in k]; print(N,wd,{q:ms([a[q] for a in x]) for q in ("spread_lab","spread_km","var_ratio","nn_ratio","dgm_ratio")})
    except Exception as e: print(N,wd,"missing",e)
print("== per-sample memorization")
for kind,d,N,wd in [(KIND,D0,N,"0") for N in (250,1000,4000,"full")]+[("linear",D0,250,"0")]:
    x=[mem[rid(kind,d,N,wd,s)] for s in range(3)]
    print(kind,N,"frac_mem",ms([a["frac_mem_final"] for a in x]),"test_below",ms([a["frac_test_below_thr_final"] for a in x]),"t_mem_ep q10/50/90",[ms([a["t_mem_epoch_q"][i] for a in x if "t_mem_epoch_q" in a]) for i in range(3)] if all("t_mem_epoch_q" in a for a in x) else "n/a",
      "rho(err_min)",ms([a["spearman_tmem"]["err_at_testmin"] for a in x if "spearman_tmem" in a]) if all("spearman_tmem" in a for a in x) else "", "pix_nn",ms([a["spearman_tmem"]["pix_nn_dist"] for a in x if "spearman_tmem" in a]) if all("spearman_tmem" in a for a in x) else "",
      "DECOUPLE tr/te rho(err,dgm)",ms([a.get("spearman_err_final_vs_dgm_final",np.nan) for a in x]),ms([a.get("spearman_te_err_final_vs_dgm_final",np.nan) for a in x]),"pl/net ratio",ms([a["pathlen_ratio"] for a in x]),ms([a["net_ratio"] for a in x]))
print("== 4. PC staging: rise spearman and slope log(rise step) vs log(lambda)")
for kind,d,N,wd in [("linear",D0,N,"0") for N in ("full",1000,4000,250)]+[(KIND,D0,N,wd) for N in ("full",1000,4000,250) for wd in ("0","1e-4")]+[(KIND,dd_,N_,"0") for dd_ in DSW for N_ in ("full",1000)]:
    sp=[];sl=[]
    for s in range(3):
        J,R=load(rid(kind,d,N,wd,s)); sp.append(J["te_pc_rise_spearman_dd"])
        d_=rd(rid(kind,d,N,wd,s)); pool=np.load(d_+"/train_index.npz")["indices"] if N!="full" else np.arange(NF)
        Xp=Xtr_all[pool].astype(np.float64); kk=min(d,32); lam=np.sort(np.linalg.eigvalsh(np.cov(Xp.T)))[::-1][:kk]
        rise=R["te_pc_rise"][:kk]; ok=np.isfinite(rise)&(rise>0)
        if ok.sum()>=4: sl.append(np.polyfit(np.log(lam[ok]),np.log(rise[ok]),1)[0])
    print(kind,d,N,wd,"spearman",ms(sp),"slope",ms(sl))
print("== 5. vfield group-affine share (R2-R1)/(1-R1) late (last quarter), test probe; lab vs km (+ krr gain)")
def vshare(R,g,s="te"):
    r1=R[f"vf_{s}_{g}_R1_affine"]; r2=R[f"vf_{s}_{g}_R2_classaffine"]; q=len(r1)*3//4; return np.nanmean((r2[q:]-r1[q:])/(1-r1[q:]))
for kind,d,N,wd in [(KIND,D0,N,"0") for N in ("full",4000,1000,250)]+[(KIND,D0,"full","1e-4"),("linear",D0,"full","0")]+[(KIND,dd_,"full","0") for dd_ in DSW]:
    L=[];K=[];R1=[]
    for s in range(3):
        J,R=load(rid(kind,d,N,wd,s)); L.append(vshare(R,"lab")); K.append(vshare(R,"km")); q=len(R["vf_te_lab_R1_affine"])*3//4; R1.append(np.nanmean(R["vf_te_lab_R1_affine"][q:]))
    print(kind,d,N,wd,"R1",ms(R1),"lab",ms(L),"km",ms(K))
for N in (("full",1000,250) if DS!="cifar10" else ()):
    L=[];K=[];R1=[]
    for s in range(3):
        J,R=load(f"{DS}_probe_none_d16_s{s}_phase_n{N}_wd0"); L.append(vshare(R,"lab")); K.append(vshare(R,"km")); q=len(R["vf_te_lab_R1_affine"])*3//4; R1.append(np.nanmean(R["vf_te_lab_R1_affine"][q:]))
    print("probe",N,"R1",ms(R1),"lab",ms(L),"km",ms(K))
print("== 6. class structure (test probe final) NCM lab/km, kNN lab/km, Fisher lab, pixel kNN lab/km, random-init NCM lab")
for name,pat in [("AE full",f"ae_{KIND}_d{D0}_s{{s}}_nfull_wd0"),("AE N1000",f"ae_{KIND}_d{D0}_s{{s}}_n1000_wd0"),("AE N250",f"ae_{KIND}_d{D0}_s{{s}}_n250_wd0"),("AE linear full",f"ae_linear_d{D0}_s{{s}}_nfull_wd0"),("probe full","probe_none_d16_s{s}_phase_nfull_wd0"),("probe N250","probe_none_d16_s{s}_phase_n250_wd0")]:
    if DS=="cifar10" and name.startswith("probe"): continue
    Js=[load(f"{DS}_"+pat.format(s=s))[0] for s in range(3)]
    print(name,{k:ms([j[k] for j in Js]) for k in ("final_te_lab_ncm","final_te_km_ncm","final_te_lab_knn","final_te_km_knn","final_te_lab_fisher","te_lab_knn_pixel","te_km_knn_pixel","initial_te_lab_ncm")})
# half-gain epoch for NCM lab/km
print("== NCM half-gain epoch (full N)")
for name,pat in [("AE",f"ae_{KIND}_d{D0}_s{{s}}_nfull_wd0"),("probe","probe_none_d16_s{s}_phase_nfull_wd0")]:
    if DS=="cifar10" and name=="probe": continue
    for g in ("lab","km"):
        e=[]
        for s in range(3):
            J,R=load(f"{DS}_"+pat.format(s=s)); y=R[f"te_{g}_ncm"]; st=R["steps"]; h=y[0]+0.5*(y[-1]-y[0]); i=np.where(y>=h)[0][0]; e.append(st[i]*128/NF)
        print(name,g,ms(e))
print("== 3b partial correlations (MLP d16, 24 runs): target log(te/tr); regress out logN, wd; N<=4000 only")
rows=[]
for N in (250,1000,4000,"full"):
    for wd in ("0","1e-4"):
        for s in range(3):
            r=rid(KIND,D0,N,wd,s); J,R=load(r); m=mem[r]["final"]; rows.append(dict(N=N,wd=float(wd),logN=np.log(NF if N=="full" else N),y=np.log(J["final_recon_test"]/J["final_recon_train"]),spread=m["spread_lab"],var=m["var_ratio"],white=m["var_ratio_white"],nn=m["nn_ratio"],effr=m["effrank_ratio"],knnov=J["final_te_knn_overlap"],cka=J["final_te_cka_pca"],er=J["final_te_effrank"],cfg=(N,wd)))
def partial(sub,key):
    X=np.c_[np.ones(len(sub)),[r["logN"] for r in sub],[r["wd"]>0 for r in sub]].astype(float)
    def res(v): v=np.array(v,float); return v-X@np.linalg.lstsq(X,v,rcond=None)[0]
    return np.corrcoef(res([r[key] for r in sub]),res([r["y"] for r in sub]))[0,1]
for lab_,sub in (("all24",rows),("N<=4000",[r for r in rows if r["N"]!="full"])):
    print(lab_,{k:round(partial(sub,k),2) for k in ("spread","var","white","nn","effr","knnov","cka","er")})
