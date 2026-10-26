import glob, numpy as np, json
R="/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/ntk"
def curve(rd):
    fs=sorted(glob.glob(f"{rd}/latents/step_*.npz")); e=[];te=[];tr=[]
    for f in fs:
        d=np.load(f); e.append(float(d['epoch']));te.append(float(d['recon_test']));tr.append(float(d['recon_train']))
    return np.array(e),np.array(te),np.array(tr)
out={}
for N in (250,1000):
    for tag,pat in (("base",f"{R}/runs/mnist_ae_mlp_d16_s{{s}}_n{N}_wd0"),("dec lr x0.1",f"{R}/lrctl/runs/mnist_ae_mlp_d16_s{{s}}_n{N}_wd0_dec0.1"),("enc lr x0.1",f"{R}/lrctl/runs/mnist_ae_mlp_d16_s{{s}}_n{N}_wd0_enc0.1")):
        rows=[]
        for s in (0,1,2):
            e,te,tr=curve(pat.format(s=s)); i=te.argmin(); fin=te[-1]
            ep_end=e[-1]
            # epoch at which gap ratio te/tr first > 1.1
            g=np.where(te/tr>1.1)[0]
            rows.append([e[i],te[i],(fin-te[i]),(fin-te[i])/te[i],tr[i],(te/tr)[np.searchsorted(e,min(e[-1],200))] if True else 0, e[g[0]] if len(g) else np.nan, ep_end, tr[-1], fin])
        a=np.array(rows); m=a.mean(0); sd=a.std(0,ddof=1)
        out[f"N{N} {tag}"]=(m,sd)
        print(f"N={N:5d} {tag:12s} t* ep {m[0]:7.1f}±{sd[0]:5.1f} | min test {m[1]:.4f}±{sd[1]:.4f} | rise(end-min) {m[2]:.4f}±{sd[2]:.4f} ({100*m[3]:.1f}%±{100*sd[3]:.1f}) | gap onset ep(te/tr>1.1) {m[6]:.1f}±{sd[6]:.1f} | final train {m[8]:.4f} test {m[9]:.4f}  [end epoch {m[7]:.0f}]")
