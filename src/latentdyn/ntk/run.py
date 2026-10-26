"""Driver: tangent-kernel coupling analysis for one dense_latent run. usage: python -m latentdyn.ntk.run <run_id> [--no_kernel]"""
import sys, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
from latentdyn.ntk import lib as L, pair as Pm
from latentdyn.model import LatentProbeModel

DL = '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent'
OUT = os.environ.get('NTK_OUT', '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/ntk_coupling')
MNIST = '/vol/bitbucket/lbastian/latent-dynamics/runs_v2/data'


def load_state(path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    names = ['encoder.net.1.weight', 'encoder.net.1.bias', 'encoder.net.3.weight', 'encoder.net.3.bias',
             'encoder.net.5.weight', 'encoder.net.5.bias', 'probe.weight', 'probe.bias']
    P = [c['model'][n].float().to(L.DEV).clone() for n in names]
    o = c['optimizer']; g = o['param_groups'][0]; st = o['state']
    if 'betas' in g:
        m = [st[i]['exp_avg'].to(L.DEV).clone() if i in st else torch.zeros_like(P[i]) for i in range(8)]
        v = [st[i]['exp_avg_sq'].to(L.DEV).clone() if i in st else torch.zeros_like(P[i]) for i in range(8)]
        step = int(st[0]['step']) if 0 in st else 0
        S = dict(kind='adam', params=P, m=m, v=v, step=step, lr=g['lr'], b1=g['betas'][0], b2=g['betas'][1], eps=g['eps'], wd=g['weight_decay'])
    else:
        m = [st[i]['momentum_buffer'].to(L.DEV).clone() if (i in st and st[i].get('momentum_buffer') is not None) else torch.zeros_like(P[i]) for i in range(8)]
        S = dict(kind='sgd', params=P, m=m, step=c['step'], lr=g['lr'], mu=g['momentum'], wd=g['weight_decay'])
    return S, c


def sim_step(ck, batch, ctx):
    """Replay one optimizer step from checkpoint dict `ck` on `batch` (full-split indices) with torch's own optimizer."""
    m = LatentProbeModel(16, 'mlp').to(L.DEV); m.load_state_dict(ck['model']); m.train()
    g = ck['optimizer']['param_groups'][0]
    if 'betas' in g:
        opt = torch.optim.Adam(m.parameters(), lr=g['lr'], betas=g['betas'], eps=g['eps'], weight_decay=g['weight_decay'])
    else:
        opt = torch.optim.SGD(m.parameters(), lr=g['lr'], momentum=g['momentum'], weight_decay=g['weight_decay'])
    import copy; opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    bi = torch.as_tensor(batch).to(L.DEV)
    opt.zero_grad(); F.cross_entropy(m(ctx['Xfull'][bi])[0], ctx['lab_full'][bi]).backward(); opt.step()
    return L.make_state(m, opt)


def main():
    rid = sys.argv[1]; kernel = '--no_kernel' not in sys.argv; ro = '--resid_only' in sys.argv
    only = [int(a) for a in sys.argv[2:] if a.lstrip('-').isdigit()]
    d = f'{DL}/{rid}'; ra = f'{DL}/_run_artifacts/runs/{rid}'
    meta = json.load(open(f'{d}/meta.json')); cs = meta['ckpt_steps']
    bidx = {}
    zhalf = {}
    for f in sorted(glob.glob(f'{d}/chunk_*.npz')):
        z = np.load(f)
        for s, lt in zip(z['steps'], z['latents_test']):
            zhalf[int(s)] = lt
    pool = np.load(f'{d}/probe_info.npz')['pool_indices']
    bidx = L.true_batches(0, pool, meta['batch_size'], meta['total_steps'])
    if os.environ.get('NTK_BATCH') == 'logged':
        import glob as _g
        bidx = {}
        for _f in sorted(_g.glob(f'{d}/chunk_*.npz')):
            _z = np.load(_f)
            for _s, _b in zip(_z['steps'], _z['batch_idx']): bidx[int(_s)] = _b[_b >= 0].astype(np.int64)  # logged batch_idx is corrupted early (see ntk_lib)
    ctx = Pm.make_ctx(MNIST, f'{ra}/probe_index.npz', f'{ra}/probe_train_index.npz', f'{ra}/train_index.npz')
    od = f'{OUT}/{rid}'; os.makedirs(od, exist_ok=True)
    pairs = [s for s in cs if s + 1 in set(cs) and s + 1 <= meta['total_steps']]
    if only: pairs = [s for s in pairs if s in only]
    for s in pairs:
        fn = f'{od}/{"resid" if ro else "pair"}_{s:07d}.json'
        if os.path.exists(fn) and not only: continue
        t0 = time.time()
        prev, ckp = load_state(f'{d}/ckpt/step_{s:07d}.pt'); nxt, ckn = load_state(f'{d}/ckpt/step_{s + 1:07d}.pt')
        b = bidx[s + 1]
        hist = [bidx[k] for k in range(s, max(s - 40, 0), -1) if k in bidx]
        obs_next = bn = None
        if s + 2 in bidx:
            nn_ = sim_step(ckn, bidx[s + 2], ctx)
            zs = [L.z_of([t for t in st['params']], ctx['Xp']) for st in (nxt, nn_)]
            obs_next, bn = zs[1] - zs[0], bidx[s + 2]
        r = Pm.analyze_pair(prev, nxt, b, ctx, obs_next=obs_next, batch_next_idx=bn, kernel=kernel, hist_batches=hist, resid_only=ro)
        # replay check: does torch's own optimizer reproduce ckpt s+1 from ckpt s + this batch?
        if ro:
            r['step'] = s + 1; json.dump(r, open(fn, 'w')); print(rid, s, 'resid', flush=True); continue
        rp = sim_step(ckp, b, ctx)
        da = torch.cat([(a - c).flatten() for a, c in zip(rp['params'], prev['params'])])
        db = torch.cat([(a - c).flatten() for a, c in zip(nxt['params'], prev['params'])])
        r['replay_cos'] = float(F.cosine_similarity(da, db, dim=0)); r['replay_relerr'] = float((da - db).norm() / db.norm().clamp_min(1e-30))
        if s in zhalf:
            zt = L.z_of(prev['params'], ctx['Xp'][:ctx['ntest']]).cpu()
            r['latent_dump_maxabs'] = float((zt - torch.as_tensor(zhalf[s]).float()).abs().max())
        r['step'] = s + 1; r['run'] = rid
        json.dump(r, open(fn, 'w'))
        print(rid, s, f'{time.time() - t0:.1f}s', 'batch_cos', round(r['batch_check']['cos'], 5), 'lin_r2', round(r['ab']['test']['lin']['r2'], 4),
              'replay', round(r['replay_cos'], 5), flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
