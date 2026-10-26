import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[k]="8"
import numpy as np, json
sys.path.insert(0,"/homes/lbastian/latent-dynamics")
import ae_analysis as A, ae_memorization as M
from scipy.stats import rankdata
R=A.AE_RUNS
cfgs={"mnist":("mnist","mlp",16),"fashion":("fashion_mnist","mlp",16),"cifar10":("cifar10","conv",64)}
def sp(a,b): return A.spearman(np.asarray(a,float),np.asarray(b,float))
def auc(m,n): # P(nonmember score > member score): higher score = non-member
    r=rankdata(np.concatenate([m,n])); return (r[len(m):].sum()-len(n)*(len(n)+1)/2)/(len(m)*len(n))
def pca_fit(X,d):
    mu=X.mean(0); U,S,Vt=np.linalg.svd(X-mu,full_matrices=False); return mu,Vt[:d]
def pca_apply(X,mu,P):
    Y=(X-mu)@P.T; res=((X-mu)**2).sum(1)-(Y**2).sum(1); return Y,np.maximum(res,1e-12)
def dgm(Z,y): return M.loo_dgm(Z,y)
def snap(rd,sub,idx):
    import glob
    fs=sorted(glob.glob(f"{rd}/{sub}/step_*.npz"));
    out=[]
    for i in idx:
        z=np.load(fs[i]); out.append((z["latents"].astype(np.float64),z["recon_err"].astype(np.float64),int(z["step"])))
    return out,len(fs)
out={}
for key,(ds,nl,d) in cfgs.items():
    Xte_all,_=A.images(ds,False); Xtr_all,_=A.images(ds,True)
    for N in (250,1000,4000):
        rec={k:[] for k in ["nl_tr","nl_te","li_tr","li_te","pca_tr","pca_te","auc_raw","auc_li","auc_pca","auc_pcaho","sp_cal_tr","sp_cal_te","sp_calho_tr","sp_calho_te","auc_nl_vs_li_err"]}
        for s in (0,1,2):
            rn=f"{R}/{ds}_ae_{nl}_d{d}_s{s}_n{N}_wd0"; rl=f"{R}/{ds}_ae_linear_d{d}_s{s}_n{N}_wd0"
            if not (os.path.isdir(rn) and os.path.isdir(rl)): continue
            ptr=np.load(f"{rn}/probe_train_index.npz")["indices"]; pte=np.load(f"{rn}/probe_index.npz")["indices"]
            tri=np.load(f"{rn}/train_index.npz")["indices"]
            assert (np.load(f"{rl}/probe_train_index.npz")["indices"]==ptr).all()
            Xtr=Xtr_all[ptr].astype(np.float64); Xte=Xte_all[pte].astype(np.float64); Xfit=Xtr_all[tri].astype(np.float64)
            hold=np.setdiff1d(np.arange(len(Xtr_all)),tri); hold=np.random.RandomState(s).choice(hold,min(10000,len(hold)),replace=False)
            mu,P=pca_fit(Xfit,d); Ytr,etr=pca_apply(Xtr,mu,P); Yte,ete=pca_apply(Xte,mu,P)
            mu2,P2=pca_fit(Xtr_all[hold].astype(np.float64),d); _,htr=pca_apply(Xtr,mu2,P2); _,hte=pca_apply(Xte,mu2,P2)
            (a,),T=snap(rn,"latents",[-1]);(b,),_=snap(rn,"latents_train",[-1]);(c,),_=snap(rl,"latents",[-1]);(e,),_=snap(rl,"latents_train",[-1])
            ytr=np.load(sorted(__import__("glob").glob(f"{rn}/latents_train/step_*.npz"))[-1])["labels"]; yte=np.load(sorted(__import__("glob").glob(f"{rn}/latents/step_*.npz"))[-1])["labels"]
            Zte,erte,_=a;Ztr,ertr,_=b;Lte,lete,_=c;Ltr,letr,_=e
            rec["nl_tr"].append(sp(ertr,dgm(Ztr,ytr)));rec["nl_te"].append(sp(erte,dgm(Zte,yte)))
            rec["li_tr"].append(sp(letr,dgm(Ltr,ytr)));rec["li_te"].append(sp(lete,dgm(Lte,yte)))
            rec["pca_tr"].append(sp(etr,dgm(Ytr,ytr)));rec["pca_te"].append(sp(ete,dgm(Yte,yte)))
            rec["auc_raw"].append(auc(ertr,erte))
            cal=lambda e,r:np.log(e)-np.log(r)
            rec["auc_li"].append(auc(cal(ertr,letr),cal(erte,lete)))
            rec["auc_pca"].append(auc(cal(ertr,etr),cal(erte,ete)))
            rec["auc_pcaho"].append(auc(cal(ertr,htr),cal(erte,hte)))
            rec["auc_nl_vs_li_err"].append(auc(letr,lete))
            rec["sp_cal_tr"].append(sp(cal(ertr,letr),dgm(Ztr,ytr)));rec["sp_cal_te"].append(sp(cal(erte,lete),dgm(Zte,yte)))
            rec["sp_calho_tr"].append(sp(cal(ertr,htr),dgm(Ztr,ytr)));rec["sp_calho_te"].append(sp(cal(erte,hte),dgm(Zte,yte)))
        o={k:(float(np.mean(v)),float(np.std(v)),len(v)) for k,v in rec.items() if v}
        out[f"{key}_{N}"]=o
        f=lambda k:"%+.2f±%.2f"%o[k][:2]
        print(key,N,"n=",o["nl_tr"][2],"| NL tr/te",f("nl_tr"),f("nl_te"),"| LIN",f("li_tr"),f("li_te"),"| PCA",f("pca_tr"),f("pca_te"),flush=True)
        print("    AUC raw %.3f | cal(linAE) %.3f | cal(PCA_N) %.3f | cal(PCA heldout) %.3f | linAE-raw %.3f"%tuple(o[k][0] for k in ["auc_raw","auc_li","auc_pca","auc_pcaho","auc_nl_vs_li_err"]),
              "| sp cal(linAE) tr/te %+.2f/%+.2f  cal(ho) %+.2f/%+.2f"%tuple(o[k][0] for k in ["sp_cal_tr","sp_cal_te","sp_calho_tr","sp_calho_te"]),flush=True)
json.dump(out,open("/tmp/claude-2292/-homes-lbastian-latent-dynamics/60dab7a5-6768-485a-b54d-fec4cb810906/scratchpad/dec.json","w"))
# evolution at N=1000
for key,(ds,nl,d) in cfgs.items():
    N=1000; rows={}
    for typ in (nl,"linear"):
        acc=[]
        for s in (0,1,2):
            rd=f"{R}/{ds}_ae_{typ}_d{d}_s{s}_n{N}_wd0"
            import glob
            T=len(glob.glob(f"{rd}/latents/step_*.npz")); idx=[int(round(q*(T-1))) for q in (0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0)]
            A_,_=snap(rd,"latents",idx);B_,_=snap(rd,"latents_train",idx)
            yte=np.load(sorted(glob.glob(f"{rd}/latents/step_*.npz"))[-1])["labels"];ytr=np.load(sorted(glob.glob(f"{rd}/latents_train/step_*.npz"))[-1])["labels"]
            g=[sp(B_[i][1],dgm(B_[i][0],ytr))-sp(A_[i][1],dgm(A_[i][0],yte)) for i in range(len(idx))]
            et=[(B_[i][1].mean(),A_[i][1].mean()) for i in range(len(idx))]
            acc.append((g,[x[0]/x[1] for x in et],[B_[i][2] for i in range(len(idx))]))
        rows[typ]=(np.mean([a[0] for a in acc],0),np.mean([a[1] for a in acc],0),acc[0][2])
    print(key,"N=1000 steps(s0)",rows[nl][2]);
    for typ in rows: print("   ",typ,"tr-te Spearman gap",np.round(rows[typ][0],2),"train/test mean err",np.round(rows[typ][1],2))
