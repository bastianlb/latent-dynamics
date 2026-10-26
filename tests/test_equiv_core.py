"""Old (baseline 71ee3eb) vs new core modules: model bitwise, metrics, config parsing. CPU only."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import _oldcode
from latentdyn import config as new_config, metrics as new_metrics, model as new_model

old_model = _oldcode.load_old("model")
old_metrics = _oldcode.load_old("metrics")
old_config = _oldcode.load_old("config")


def _eq(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _eq(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert (a == b) or (a != a and b != b), f"{path}: {a} != {b}"


CASES = [
    dict(latent_dim=16, encoder_arch="mlp", objective="probe", in_channels=1, img_size=28),
    dict(latent_dim=16, encoder_arch="cnn", objective="probe", in_channels=1, img_size=28),
    dict(latent_dim=16, encoder_arch="resnet18", objective="probe", in_channels=3, img_size=32),
    dict(latent_dim=16, encoder_arch="mlp", objective="autoencoder", in_channels=1, img_size=28),
    dict(latent_dim=8, encoder_arch="convae", objective="autoencoder", in_channels=3, img_size=32),
    dict(latent_dim=8, encoder_arch="mlp", objective="autoencoder", in_channels=1, img_size=28, linear_ae=True),
]


def test_model_bitwise():
    for kw in CASES:
        torch.manual_seed(0); mo = old_model.LatentProbeModel(**kw)
        torch.manual_seed(0); mn = new_model.LatentProbeModel(**kw)
        so, sn = mo.state_dict(), mn.state_dict()
        assert so.keys() == sn.keys()
        for k in so:
            assert torch.equal(so[k], sn[k]), (kw, k)
        mo.eval(); mn.eval()
        x = torch.randn(4, kw["in_channels"], kw["img_size"], kw["img_size"], generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            oo, on = mo(x), mn(x)
        for a, b in zip(oo, on):
            if a is None:
                assert b is None
            else:
                assert torch.equal(a, b), kw


def test_metrics_equal():
    rng = np.random.RandomState(0)
    lat = rng.randn(300, 16); lab = np.repeat(np.arange(10), 30)
    lat += lab[:, None] * 0.3
    for name in ("pairwise_distances", "normalize_latents", "fisher_separation_ratio", "global_distance_stats",
                 "intra_class_compactness", "inter_class_separation", "class_centroids"):
        args = (lat,) if name in ("pairwise_distances", "normalize_latents", "global_distance_stats") else (lat, lab)
        _eq(getattr(old_metrics, name)(*args), getattr(new_metrics, name)(*args), name)
    _eq(old_metrics.compute_all_metrics(lat, lab), new_metrics.compute_all_metrics(lat, lab), "compute_all_metrics")
    _eq(old_metrics.pmag(lat), new_metrics.pmag(lat), "pmag")


FLAGSETS = [
    [],
    ["--dataset", "mnist", "--encoder_arch", "mlp", "--objective", "probe", "--latent_dim", "16", "--seed", "3"],
    ["--dataset", "cifar10", "--encoder_arch", "convae", "--objective", "autoencoder", "--latent_norm", "none",
     "--optimizer", "adam", "--lr", "1e-3", "--weight_decay", "1e-4", "--train_subset", "1000", "--seed", "1"],
    ["--dataset", "fashion_mnist", "--objective", "autoencoder", "--linear_ae", "--probe_train", "--train_eval", "full",
     "--isolate_snapshot_rng", "--log_schedule", "log", "--run_id", "foo", "--artifacts_root", "/tmp/x"],
]


def test_config_equal():
    for argv in FLAGSETS:
        co, cn = old_config.config_from_args(argv), new_config.config_from_args(argv)
        _eq(co.to_json_dict(), cn.to_json_dict(), str(argv))
        po, pn = old_config.get_paths(co), new_config.get_paths(cn)
        _eq({k: str(v) for k, v in vars(po).items()}, {k: str(v) for k, v in vars(pn).items()}, "paths")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
