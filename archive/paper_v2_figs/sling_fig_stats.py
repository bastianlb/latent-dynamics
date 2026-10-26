"""Numbers quoted in the fig_sling caption (n=1000, wd=0, seed 0): latent norm and Fisher of Adam vs SGD."""
from common import *
runs = cache_runs()
for tag in ["sl40k_n1000_wd0", "sgd40k_n1000_wd0"]:
    d = np.load(runs[tag][0]); s = d["step"]; nm = d["test/norm"]; f = d["test/fisher"]
    i1k = int(np.argmin(np.abs(s - 1000)))
    print(tag, f"norm step{int(s[i1k])}={nm[i1k]:.1f} final={nm[-1]:.1f} peak={nm.max():.1f} (step {int(s[np.argmax(nm)])}) final/step1000={nm[-1]/nm[i1k]:.2f} peak/step1000={nm.max()/nm[i1k]:.2f}")
    print("   fisher max-before-last-event / final:", f"max={np.nanmax(f):.2f} (step {int(s[np.nanargmax(f)])}) final={f[-1]:.2f}")
