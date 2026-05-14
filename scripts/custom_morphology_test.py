"""
OPV 2D PINN — Morphology Study with Fixed Config Parameters
============================================================
Uses fixed config.txt material parameters across all morphologies.
The only thing varying is the morphology structure (spatial arrangement
of donor and acceptor domains).

Scientific question: How does morphology structure alone affect Jsc,
given fixed material properties?

What this script does:
  1. Load N morphologies from chem_morph_data (spread across dataset)
  2. Train a PINN for each using FIXED config.txt params
  3. Record Jsc, interface density, donor fraction, domain size
  4. Produce a rich visualisation showing morphology → Jsc relationships

Config.txt params used (fixed for ALL morphologies):
  mu_n = 2e-7  mu_p = 1.5e-7  mu_x = 3.9e-9
  tau_x = 1e-6  kdiss_factor = 1.0  Gx = 1e28

Run:
    python opv_pinn_morphstudy.py --data_dir /path/to/data --n_morphs 10
    python opv_pinn_morphstudy.py --data_dir /path/to/data --n_morphs 20 --spread jsc
    python opv_pinn_morphstudy.py --data_dir /path/to/data --morph_indices 0,100,500,1000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import math, os, argparse
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label, gaussian_filter

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# SECTION 1 — FIXED CONFIG.TXT PARAMETERS
# These never change regardless of which morphology we train on
# =============================================================================
class ConfigParams:
    """
    Fixed material parameters from config.txt.
    Used for ALL morphologies — only morphology structure varies.
    """
    q    = 1.602e-19; kb = 1.381e-23; eps0 = 8.854e-12
    T    = 300.0; eps_D = 3.0; eps_A = 3.9; E_g = 1.1
    Height = 100e-9; muRatio = 0.01; Gx = 1e28

    # Config.txt material params — FIXED
    mu_n         = 2e-7
    mu_p         = 1.5e-7
    mu_x         = 3.9e-9
    tau_x        = 1e-6
    kdiss_factor = 1.0

    def __init__(self):
        self.VT      = self.kb*self.T/self.q
        self.eps_avg = (self.eps_D+self.eps_A)/2
        self.eps_si  = self.eps_avg*self.eps0
        self.V_bi    = self.E_g - 2*self.VT
        self.Dn      = self.mu_n*self.VT
        self.Dp      = self.mu_p*self.VT
        self.Dx      = self.mu_x*self.VT
        self.kx      = 1.0/self.tau_x
        self.k_diss  = self.kdiss_factor*1e6
        k_rec_L      = self.q*(self.mu_n+self.mu_p)/self.eps_si
        self.k_rec   = 0.1*k_rec_L

        print("Fixed config.txt parameters:")
        print(f"  mu_n={self.mu_n:.1e}  mu_p={self.mu_p:.1e}  "
              f"mu_x={self.mu_x:.1e}  tau_x={self.tau_x:.1e}")
        print(f"  kdiss_factor={self.kdiss_factor:.1f}  Gx={self.Gx:.1e}")
        print(f"  V_bi={self.V_bi:.4f} V")


# =============================================================================
# SECTION 2 — NON-DIMENSIONALISATION (fixed params -> simple clean norms)
# =============================================================================
class NormParams:
    def __init__(self, p: ConfigParams):
        self.L0    = p.Height
        self.V0    = p.E_g
        self.VT_nd = p.VT/p.E_g

        _J_REF     = 3e21*(2e-7*p.VT)*p.q/p.Height
        self.n0    = _J_REF/(p.Dn*p.q/p.Height)
        self.tau0  = p.Height**2/p.Dn

        self.mu_n_nd = 1.0
        self.mu_p_nd = p.mu_p/p.mu_n          # = 0.75 for config params
        self.mu_x_nd = p.Dx/p.Dn
        self.muRatio = p.muRatio

        self.Gx_nd    = p.Gx*self.tau0/self.n0
        self.kx_nd    = p.kx*self.tau0
        self.kdiss_nd = p.k_diss*self.tau0
        self.Gx_scale = min(1.0, self.kx_nd/self.Gx_nd) if self.Gx_nd>1e-12 else 1.0
        self.loss_scale = float(np.clip(max(1.0, self.kx_nd/5.0), 1.0, 20.0))

        V_bi_nd   = p.V_bi/p.E_g
        Gx_eff    = self.Gx_nd*self.Gx_scale
        X_avg     = Gx_eff/(self.kx_nd+self.kdiss_nd*V_bi_nd+1e-12)
        D_avg     = self.kdiss_nd*X_avg*V_bi_nd
        self.krec_nd  = D_avg/0.25
        p.k_rec       = self.krec_nd/(self.n0*self.tau0)

        self.debye_ratio = p.q*self.n0*p.Height**2/(p.eps_si*p.E_g)
        self.V_bi_nd     = V_bi_nd
        self.J_scale     = self.n0*p.Dn*p.q/p.Height
        # With config params mu_p_nd=0.75 < 1, so J_scale_phys = J_scale (no correction)
        self.J_scale_phys = self.J_scale / max(1.0, self.mu_p_nd)
        self.n_min        = max(1e-4, math.exp(-V_bi_nd/(2*self.VT_nd)))

        print(f"\n  NormParams (config.txt, fixed for all morphologies):")
        print(f"    kx_nd={self.kx_nd:.4f}  kdiss_nd={self.kdiss_nd:.4f}  "
              f"krec_nd={self.krec_nd:.4f}")
        print(f"    Gx_scale={self.Gx_scale:.4f}  n_min={self.n_min:.2e}")
        print(f"    J_scale={self.J_scale_phys*0.1:.4f} mA/cm2/J_nd\n")


# =============================================================================
# SECTION 3 — DATA LOADING
# =============================================================================
def load_data(data_dir):
    path = os.path.join(data_dir, 'chem_morph_data.npy')
    print(f"Loading {path}...")
    raw  = np.load(path, allow_pickle=True)
    N    = len(raw)
    morphs = np.zeros((N, 128, 128), dtype=np.float32)
    params = np.zeros((N, 5),        dtype=np.float32)
    jsc    = np.zeros(N,             dtype=np.float32)
    for i, row in enumerate(raw):
        morphs[i] = np.array(row[0], dtype=np.float32)
        params[i] = np.array(row[1], dtype=np.float32)
        jsc[i]    = float(row[2])
        if i % 15000 == 0: print(f"  {i}/{N}")
    print(f"Loaded {N} samples | GT Jsc [{jsc.min():.3f},{jsc.max():.3f}] mA/cm²\n")
    return morphs, params, jsc


def select_morphologies(jsc_gt, n_morphs, spread='uniform', rng_seed=42):
    """
    Select morphology indices to study.
    spread='uniform' : evenly spaced across dataset indices
    spread='jsc'     : spread across the GT Jsc range (low to high)
    spread='random'  : random selection
    """
    N   = len(jsc_gt)
    rng = np.random.default_rng(rng_seed)
    if spread == 'jsc':
        # Pick morphologies spread across the Jsc range
        pcts  = np.linspace(5, 95, n_morphs)
        thresholds = np.percentile(jsc_gt, pcts)
        indices = [int(np.argmin(np.abs(jsc_gt - t))) for t in thresholds]
        # Remove duplicates
        seen = set(); indices = [i for i in indices if not (i in seen or seen.add(i))]
    elif spread == 'random':
        indices = rng.choice(N, size=n_morphs, replace=False).tolist()
    else:  # uniform
        indices = list(np.linspace(0, N-1, n_morphs, dtype=int))
    print(f"Selected {len(indices)} morphologies (spread='{spread}'): {indices}")
    return indices


# =============================================================================
# SECTION 4 — MORPHOLOGY METRICS
# Compute structural features from the 128x128 phase grid
# =============================================================================
def compute_morph_metrics(morph_grid):
    """
    Returns a dict of structural metrics for one morphology.
    All metrics are physically meaningful predictors of device performance.
    """
    mg = morph_grid
    donor_mask    = mg > 0.5
    acceptor_mask = mg <= 0.5

    donor_frac    = float(donor_mask.mean())

    # Interface density: magnitude of morphology gradient
    gx = np.abs(np.gradient(mg, axis=0))
    gy = np.abs(np.gradient(mg, axis=1))
    interface_density = float((gx + gy).mean())

    # Domain size: average size of connected donor regions (pixels)
    labeled, n_domains = label(donor_mask)
    if n_domains > 0:
        domain_sizes = [np.sum(labeled == i) for i in range(1, n_domains+1)]
        avg_domain_size = float(np.mean(domain_sizes))
        max_domain_size = float(np.max(domain_sizes))
    else:
        avg_domain_size = max_domain_size = 0.0

    # Percolation: does donor phase span from x=0 to x=1?
    # Check if there is a connected donor path from left edge to right edge
    left_donors  = set(zip(*np.where(donor_mask[:5,  :])))
    right_donors = set(zip(*np.where(donor_mask[-5:, :])))
    # Simplified percolation check using labeled regions
    donor_perc = False
    for i in range(1, n_domains+1):
        region = labeled == i
        if region[:5, :].any() and region[-5:, :].any():
            donor_perc = True; break

    # Phase smoothness: standard deviation of phase field
    phase_std = float(mg.std())

    return {
        'donor_frac'       : donor_frac,
        'interface_density': interface_density,
        'avg_domain_size'  : avg_domain_size / (128*128),  # normalised
        'max_domain_size'  : max_domain_size / (128*128),
        'n_donor_domains'  : n_domains,
        'donor_percolation': donor_perc,
        'phase_std'        : phase_std,
    }


# =============================================================================
# SECTION 5 — MORPHOLOGY HANDLER
# =============================================================================
class MorphologyHandler:
    def __init__(self, grid, norm):
        H, W = grid.shape
        self.interp = RegularGridInterpolator(
            (np.linspace(0,1,H), np.linspace(0,1,W)),
            grid, method='linear', bounds_error=False, fill_value=0.5)
        self.norm = norm; self.morph_grid = grid
        gx = np.abs(np.gradient(grid, axis=0))
        gy = np.abs(np.gradient(grid, axis=1))
        self.metrics = compute_morph_metrics(grid)
        print(f"  Morph {H}x{W}  donor={self.metrics['donor_frac']:.3f}  "
              f"intf={self.metrics['interface_density']:.4f}  "
              f"n_domains={self.metrics['n_donor_domains']}")

    def _phase(self, xy):
        p = self.interp(xy.detach().cpu().numpy()).astype(np.float32)
        return torch.sigmoid(10*(torch.tensor(p).unsqueeze(1).to(xy.device)-0.5))

    def get_Gx(self, xy):
        return self.norm.Gx_nd*self.norm.Gx_scale*self._phase(xy)

    def get_mu_n(self, xy):
        d = self._phase(xy)
        return self.norm.mu_n_nd*((1-d)+self.norm.muRatio*d)

    def get_mu_p(self, xy):
        d = self._phase(xy)
        return self.norm.mu_p_nd*(d+self.norm.muRatio*(1-d))


# =============================================================================
# SECTION 6 — NETWORK
# =============================================================================
class FieldNet(nn.Module):
    def __init__(self, nh=5, nn_=96):
        super().__init__()
        layers = [nn.Linear(2, nn_), nn.Tanh()]
        for _ in range(nh-1): layers += [nn.Linear(nn_, nn_), nn.Tanh()]
        layers.append(nn.Linear(nn_, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, xy): return self.net(xy)


class OPV_PINN_2D(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm; self.n_min = norm.n_min
        self.net_phi = FieldNet(); self.net_n = FieldNet()
        self.net_p   = FieldNet(); self.net_X = FieldNet()

    def forward(self, xy):
        x = xy[:, 0:1]; nm = self.n_min
        phi = (self.norm.V_bi_nd/2)*(1-x)+(-self.norm.V_bi_nd/2)*x \
              +x*(1-x)*self.net_phi(xy)
        n = nm*(1-x) + 1.0*x + x*(1-x)*F.softplus(self.net_n(xy))
        p = 1.0*(1-x) + nm*x + x*(1-x)*F.softplus(self.net_p(xy))
        X = x*(1-x)*F.softplus(self.net_X(xy)+1.0)
        return phi, n, p, X


# =============================================================================
# SECTION 7 — RESIDUALS AND LOSS
# =============================================================================
def g2(f, xy):
    g = torch.autograd.grad(f, xy, torch.ones_like(f),
                             create_graph=True, retain_graph=True)[0]
    return g[:,0:1], g[:,1:2]


def residuals(model, xy, norm, morph):
    xy = xy.requires_grad_(True)
    phi, n, p, X = model(xy)
    px, py = g2(phi, xy); nx, ny = g2(n, xy)
    qx, qy = g2(p,   xy); Xx, Xy = g2(X, xy)
    pxx, _ = g2(px, xy); _, pyy = g2(py, xy)
    Xxx, _ = g2(Xx, xy); _, Xyy = g2(Xy, xy)
    Lphi = pxx+pyy; LX = Xxx+Xyy
    Em  = torch.sqrt(px**2+py**2+1e-8)
    Gl  = morph.get_Gx(xy)
    mn  = morph.get_mu_n(xy); mp = morph.get_mu_p(xy)
    D   = norm.kdiss_nd*X*Em; R = norm.krec_nd*n*p
    Jnx = mn*(n*px+nx); Jny = mn*(n*py+ny)
    Jpx = mp*(p*px-qx); Jpy = mp*(p*py-qy)
    dJnx, _ = g2(Jnx, xy); _, dJny = g2(Jny, xy)
    dJpx, _ = g2(Jpx, xy); _, dJpy = g2(Jpy, xy)
    rP = Lphi - norm.debye_ratio*(n-p)
    rn = (dJnx+dJny) - R + D
    rp = -(dJpx+dJpy) - R + D
    rX = norm.mu_x_nd*LX - norm.kx_nd*X - D + Gl
    return rP, rn, rp, rX, Jnx+Jpx


def get_w(ep, N, s):
    f = ep/N
    if f < 0.20: return {'P':20*s,'n':0.01,'p':0.01,'X':0.1*s,'Jc':0.0}
    if f < 0.60: return {'P': 5*s,'n': 0.1,'p': 0.1,'X': 0.5, 'Jc':0.0}
    return              {'P': 1*s,'n': 1.0,'p': 1.0,'X': 1.0, 'Jc':10.0}


def loss_fn(model, xy, norm, morph, w):
    rP, rn, rp, rX, Jt = residuals(model, xy, norm, morph)
    # Interface mask: peaks at 1.0 where phase=0.5 (donor-acceptor boundary)
    # 1x weight in bulk, 4x weight at interfaces
    phase_vals  = morph._phase(xy).detach()
    intf_mask   = 4.0 * phase_vals * (1.0 - phase_vals)   # in [0, 1]
    intf_weight = 1.0 + 3.0 * intf_mask                   # in [1, 4]

    lP = w['P'] * (rP**2 * intf_weight).mean()
    ln = w['n'] * (rn**2 * intf_weight).mean()
    lp = w['p'] * (rp**2 * intf_weight).mean()
    lX = w['X'] * (rX**2 * intf_weight).mean()

    if w['Jc'] > 0:
        xy2 = xy.detach().requires_grad_(True)
        phi2, n2, p2, _ = model(xy2)
        px2, _ = g2(phi2, xy2); nx2, _ = g2(n2, xy2); qx2, _ = g2(p2, xy2)
        Jt2 = norm.mu_n_nd*(n2*px2+nx2) + norm.mu_p_nd*(p2*px2-qx2)
        lJc = w['Jc']*(Jt2.var()+0.5*((Jt2-Jt2.mean().detach())**2).mean())
    else:
        lJc = torch.tensor(0.0, device=xy.device)

    total = lP+ln+lp+lX+lJc
    return total, {'total':total.item(),'P':lP.item(),
                   'n':ln.item(),'p':lp.item(),'X':lX.item(),'Jc':lJc.item()}


# =============================================================================
# SECTION 8 — COLLOCATION POINTS
# =============================================================================
def colloc(morph_handler, n_bulk=1000, n_intf=300):
    xy_b = torch.rand(n_bulk, 2)
    mg = morph_handler.morph_grid; H, W = mg.shape
    grad_mag = np.abs(np.gradient(mg,axis=0))+np.abs(np.gradient(mg,axis=1))
    prob = (grad_mag**2).flatten() 
    if prob.sum() > 0:
        prob /= prob.sum()
        idx = np.random.choice(H*W, size=n_intf, p=prob, replace=True)
        xi  = (idx//W/H + np.random.randn(n_intf)*0.01).clip(0.01, 0.99)
        yi  = (idx%W/W  + np.random.randn(n_intf)*0.01).clip(0.01, 0.99)
        xy_i = torch.tensor(np.stack([xi,yi], axis=1), dtype=torch.float32)
    else:
        xy_i = torch.rand(n_intf, 2)
    n_bl = 200
    xa   = np.random.uniform(0.001, 0.015, n_bl)
    xc   = np.random.uniform(0.985, 0.999, n_bl)
    ybl  = np.random.rand(n_bl*2)
    xy_bl = torch.tensor(np.stack([np.concatenate([xa,xc]),ybl], axis=1),
                          dtype=torch.float32)
    return torch.cat([xy_b, xy_i, xy_bl], dim=0).to(DEVICE)


# =============================================================================
# SECTION 9 — TRAINING
# =============================================================================
def train(model, norm, morph, name="", n_epochs=20000, lr=5e-4, pe=4000):
    xy  = colloc(morph)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    def lr_lambda(ep):
        w = 500
        if ep < w: return ep/w
        return 0.5*(1+math.cos(math.pi*(ep-w)/(n_epochs-w)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    hist = {k:[] for k in ['total','P','n','p','X','Jc']}

    print(f"\n  Training {name} | {n_epochs} epochs | {len(xy)} pts")
    print(f"  {'Ep':>6}  {'Total':>9}  {'P':>9}  {'n':>8}  {'p':>8}  "
          f"{'X':>8}  {'Jc':>8}")
    print(f"  {'-'*62}")

    for ep in range(n_epochs):
        opt.zero_grad()
        loss, losses = loss_fn(model, xy, norm, morph,
                                get_w(ep, n_epochs, norm.loss_scale))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step(); sch.step()
        for k, v in losses.items(): hist[k].append(v)
        if ep % pe == 0 or ep == n_epochs-1:
            print(f"  {ep:>6}  {losses['total']:>9.3e}  {losses['P']:>9.3e}  "
                  f"{losses['n']:>8.3e}  {losses['p']:>8.3e}  "
                  f"{losses['X']:>8.3e}  {losses['Jc']:>8.3e}")
    return hist


# =============================================================================
# SECTION 10 — JSC COMPUTATION
# =============================================================================
def compute_jsc(model, norm, n_pts=300):
    model.eval()
    Jsc_vals = []
    for x_val in [0.005, 0.995]:
        xy = torch.cat([torch.full((n_pts,1), x_val),
                        torch.linspace(0.01,0.99,n_pts).unsqueeze(1)],
                       dim=1).to(DEVICE).requires_grad_(True)
        phi, n, p, X = model(xy)
        gp  = torch.autograd.grad(phi, xy, torch.ones_like(phi),
                                   create_graph=False, retain_graph=True)[0]
        gn  = torch.autograd.grad(n,   xy, torch.ones_like(n),
                                   create_graph=False, retain_graph=True)[0]
        gpp = torch.autograd.grad(p,   xy, torch.ones_like(p),
                                   create_graph=False)[0]
        Jn  = norm.mu_n_nd*(n*gp[:,0:1]+gn[:,0:1])
        Jp  = norm.mu_p_nd*(p*gp[:,0:1]-gpp[:,0:1])
        Jt  = (Jn+Jp).detach().cpu().numpy().flatten()
        Jsc_vals.append(abs(float(np.mean(Jt))))
    Jsc    = float(np.mean(Jsc_vals))*norm.J_scale_phys*0.1
    latvar = float(np.std(Jsc_vals) / (np.mean(Jsc_vals)+1e-12))
    print(f"  Jsc anode={Jsc_vals[0]*norm.J_scale_phys*0.1:.4f}  "
          f"cathode={Jsc_vals[1]*norm.J_scale_phys*0.1:.4f}  "
          f"avg={Jsc:.4f} mA/cm2")
    return Jsc, latvar


# =============================================================================
# SECTION 11 — GRID EVALUATION
# =============================================================================
def eval_grid(model, norm, morph, gs=64):
    model.eval()
    XX, YY = np.meshgrid(np.linspace(0.01,0.99,gs),
                          np.linspace(0.01,0.99,gs), indexing='ij')
    flat = torch.tensor(np.stack([XX.flatten(),YY.flatten()],axis=1),
                        dtype=torch.float32).to(DEVICE)
    pl, nl, ppl, Xl, Jl = [], [], [], [], []
    rP_l, rn_l, rp_l, rX_l = [], [], [], []
    for i in range(0, len(flat), 512):
        b = flat[i:i+512].requires_grad_(True)
        ph, nb, pb, Xb = model(b)
        
        rP, rn, rp, rX, Jt = residuals(model, b, norm, morph)

        pl.append(ph.detach().cpu().numpy())
        nl.append(nb.detach().cpu().numpy())
        ppl.append(pb.detach().cpu().numpy())
        Xl.append(Xb.detach().cpu().numpy())
        Jl.append(Jt.detach().cpu().numpy())
        rP_l.append(rP.detach().cpu().numpy())
        rn_l.append(rn.detach().cpu().numpy())
        rp_l.append(rp.detach().cpu().numpy())
        rX_l.append(rX.detach().cpu().numpy())
    r = lambda l: np.concatenate(l).reshape(gs,gs)
    return r(pl), r(nl), r(ppl), r(Xl), r(Jl), r(rP_l), r(rn_l), r(rp_l), r(rX_l)


# =============================================================================
# SECTION 12 — INDIVIDUAL MORPHOLOGY PLOT (fields + loss)
# =============================================================================
def plot_single(morph, phi, n, p, X, Jx, rP, rn, rp, rX, hist, Jsc, metrics,
                morph_idx, norm, save_dir):
    fig = plt.figure(figsize=(18, 8))
    gs_ = gridspec.GridSpec(2, 5, hspace=0.4, wspace=0.35)
    fig.suptitle(
        f'Morphology #{morph_idx}  —  Physical Meaningfulness Check\n'
        f'Final Loss = {hist["total"][-1]:.3e} | PINN Jsc = {Jsc:.4f} mA/cm²\n'
        f'donor={metrics["donor_frac"]:.3f}  '
        f'intf={metrics["interface_density"]:.4f}  '
        f'domains={metrics["n_donor_domains"]}',
        fontsize=11, fontweight='bold'
    )
    mg  = morph.morph_grid; ext = [0, 100, 0, 100]

    def ct(ax):
        ax.contour(np.linspace(0,100,mg.shape[0]),
                   np.linspace(0,100,mg.shape[1]),
                   mg.T, levels=[0.5], colors='white', linewidths=0.7, alpha=0.6)

    def hm(ax, d, t, c, vn=None, vx=None):
        im = ax.imshow(d.T, origin='lower', cmap=c, extent=ext, aspect='auto',
                       vmin=vn, vmax=vx)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ct(ax)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel('x (nm)', fontsize=8); ax.set_ylabel('y (nm)', fontsize=8)
        ax.tick_params(labelsize=7)

    def hm_res(ax, d, t):
        vm = max(abs(d.min()), abs(d.max())) + 1e-12
        hm(ax, d, t, 'coolwarm', -vm, vm)

    ax0 = fig.add_subplot(gs_[0,0])
    im  = ax0.imshow(mg.T, origin='lower', cmap='RdYlBu',
                     extent=ext, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title('Morphology\nred=donor  blue=acceptor', fontsize=9)
    ax0.set_xlabel('x (nm)', fontsize=8); ax0.set_ylabel('y (nm)', fontsize=8)

    hm(fig.add_subplot(gs_[0,1]), phi*norm.V0, 'φ (V)', 'RdBu_r')
    hm(fig.add_subplot(gs_[0,2]), n, 'Electron n', 'Greens')
    hm(fig.add_subplot(gs_[0,3]), p, 'Hole p', 'Oranges')
    hm(fig.add_subplot(gs_[0,4]), X, 'Exciton X', 'YlOrBr')

    # Loss curves
    ax_loss = fig.add_subplot(gs_[1,0])
    for k,c,lb in [('total','k','total'),('P','#3B8BD4','Poisson'),
                    ('n','#1D9E75','n'),('p','#E85D24','p'),
                    ('X','#BA7517','X'),('Jc','#534AB7','Jcons')]:
        if k in hist and any(v>0 for v in hist[k]):
            ax_loss.semilogy(hist[k], color=c, lw=1.5 if k=='total' else 1, label=lb)
    ax_loss.axhline(0.01, color='gray', lw=1, ls='--', alpha=0.5)
    ax_loss.set_xlabel('Epoch', fontsize=8); ax_loss.set_ylabel('Loss', fontsize=8)
    ax_loss.set_title('Training loss', fontsize=9); ax_loss.legend(fontsize=7)
    ax_loss.spines['top'].set_visible(False); ax_loss.spines['right'].set_visible(False)

    hm_res(fig.add_subplot(gs_[1,1]), rP, 'Poisson Residual (rP)')
    hm_res(fig.add_subplot(gs_[1,2]), rn, 'Electron Residual (rn)')
    hm_res(fig.add_subplot(gs_[1,3]), rp, 'Hole Residual (rp)')
    hm_res(fig.add_subplot(gs_[1,4]), rX, 'Exciton Residual (rX)')

    fname = os.path.join(save_dir, f'morph_study_{morph_idx}.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# SECTION 13 — SUMMARY VISUALISATION (all morphologies together)
# =============================================================================
def plot_summary(results, save_dir):
    """
    Four-panel summary:
      1. Morphology gallery (thumbnails sorted by PDE Loss)
      2. Interface density vs Final Loss scatter
      3. Donor fraction vs Final Loss scatter
      4. Bar chart of Final Loss ranked by morphology
    """
    results_sorted = sorted(results, key=lambda r: r['loss'])
    N = len(results_sorted)

    fig = plt.figure(figsize=(20, 14))
    gs_ = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)
    fig.suptitle(
        f'OPV Morphology Study — Physical Meaningfulness\n'
        f'Loss range: [{min(r["loss"] for r in results):.3e}, '
        f'{max(r["loss"] for r in results):.3e}]',
        fontsize=13, fontweight='bold'
    )

    # Panel 1: Gallery of morphology thumbnails sorted by Loss
    ax_gal = fig.add_subplot(gs_[0, :])
    cols   = min(N, 10)
    rows   = math.ceil(N / cols)
    inner  = gridspec.GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=gs_[0, :], hspace=0.1, wspace=0.05)
    for i, r in enumerate(results_sorted):
        ax = fig.add_subplot(inner[i//cols, i%cols])
        ax.imshow(r['morph'].T, origin='lower', cmap='RdYlBu', vmin=0, vmax=1)
        ax.set_title(f"#{r['idx']}\nL={r['loss']:.2e}", fontsize=6)
        ax.axis('off')
    ax_gal.axis('off')
    ax_gal.set_title('Morphologies sorted by Final PDE Loss (low → high)',
                      fontsize=10, fontweight='bold', pad=2)

    # Panel 2: Interface density vs Loss
    ax2 = fig.add_subplot(gs_[1, 0])
    intf  = [r['metrics']['interface_density'] for r in results]
    losses = [r['loss'] for r in results]
    idxs  = [r['idx'] for r in results]
    sc = ax2.scatter(intf, losses, s=80, c=losses, cmap='viridis',
                     edgecolors='gray', lw=0.5, zorder=3)
    plt.colorbar(sc, ax=ax2, label='Final PDE Loss')
    for i, (x, y, idx) in enumerate(zip(intf, losses, idxs)):
        ax2.annotate(f'#{idx}', (x, y), fontsize=7,
                     xytext=(3, 3), textcoords='offset points')
    if len(intf) > 2:
        z = np.polyfit(intf, losses, 1)
        xr = np.linspace(min(intf), max(intf), 100)
        ax2.plot(xr, np.polyval(z, xr), 'k--', lw=1.2, alpha=0.6)
        from scipy.stats import pearsonr
        r_val, _ = pearsonr(intf, losses)
        ax2.set_title(f'Interface density vs Final Loss  (r={r_val:.3f})', fontsize=10)
    else:
        ax2.set_title('Interface density vs Final Loss', fontsize=10)
    ax2.set_xlabel('Interface density', fontsize=10)
    ax2.set_ylabel('Final PDE Loss', fontsize=10)
    ax2.set_yscale('log')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # Panel 3: Ranked bar chart by Loss
    ax3 = fig.add_subplot(gs_[1, 1])
    loss_sorted  = [r['loss'] for r in results_sorted]
    idx_sorted  = [r['idx'] for r in results_sorted]
    colors      = plt.cm.viridis(np.linspace(0.9, 0.2, N))
    bars = ax3.barh(range(N), loss_sorted, color=colors, alpha=0.85,
                    edgecolor='white')
    ax3.set_yticks(range(N))
    ax3.set_yticklabels([f'#{i}' for i in idx_sorted], fontsize=8)
    ax3.set_xlabel('Final PDE Loss', fontsize=10)
    ax3.set_title('PDE Loss ranking across morphologies',
                  fontsize=10)
    ax3.set_xscale('log')
    for bar, val in zip(bars, loss_sorted):
        ax3.text(val * 1.1, bar.get_y()+bar.get_height()/2,
                 f'{val:.2e}', va='center', fontsize=7)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    fname = os.path.join(save_dir, 'morphology_study_summary.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved summary: {fname}")


def plot_correlation_matrix(results, save_dir):
    """
    Correlation matrix between all structural metrics and Final Loss.
    Shows which morphology features predict physical consistency.
    """
    metric_keys = ['donor_frac', 'interface_density', 'avg_domain_size',
                   'n_donor_domains', 'phase_std']
    metric_labels = ['Donor\nfraction', 'Interface\ndensity', 'Avg domain\nsize',
                     'N donor\ndomains', 'Phase\nstd']

    losses = np.array([r['loss'] for r in results])
    data = np.column_stack([
        [r['metrics'][k] for r in results] for k in metric_keys
    ] + [losses])
    labels = metric_labels + ['Final\nLoss']

    from scipy.stats import pearsonr
    n = len(labels)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if len(data) > 2:
                corr[i,j], _ = pearsonr(data[:,i], data[:,j])
            else:
                corr[i,j] = 1.0 if i==j else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(corr[i,j])>0.5 else 'black')
    ax.set_title('Correlation: morphology metrics vs Final PDE Loss\n(config.txt params)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(save_dir, 'morphology_correlation.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved correlations: {fname}")


# =============================================================================
# SECTION 14 — MAIN STUDY LOOP
# =============================================================================
def run_study(morphs, jsc_gt, indices, norm, n_epochs, save_dir):
    """
    Run PINN for each selected morphology, collect results.
    Note: jsc_gt is dataset GT (different params) — used only for reference,
    not for error computation since we use config.txt params.
    """
    results = []

    for i, idx in enumerate(indices):
        gt_ref = float(jsc_gt[idx])  # reference only — different params
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(indices)}] Morphology #{idx}  "
              f"(dataset GT ref = {gt_ref:.4f} mA/cm² — different params)")
        print(f"{'='*60}")

        morph  = MorphologyHandler(morphs[idx], norm)
        model  = OPV_PINN_2D(norm).to(DEVICE)
        hist   = train(model, norm, morph, name=f"m{idx}",
                       n_epochs=n_epochs, lr=5e-4)
        Jsc, latvar = compute_jsc(model, norm)

        phi2, n2, p2, X2, Jx2, rP2, rn2, rp2, rX2 = eval_grid(model, norm, morph, gs=64)
        plot_single(morph, phi2, n2, p2, X2, Jx2, rP2, rn2, rp2, rX2, hist, Jsc,
                    morph.metrics, idx, norm, save_dir)

        results.append({
            'idx'     : idx,
            'jsc'     : Jsc,
            'gt_ref'  : gt_ref,
            'latvar'  : latvar,
            'loss'    : hist['total'][-1],
            'metrics' : morph.metrics,
            'morph'   : morphs[idx],
        })

        print(f"  PINN Jsc = {Jsc:.4f} mA/cm²  "
              f"latvar={latvar:.4f}  loss={hist['total'][-1]:.3e}")

    # Print summary table
    print(f"\n{'='*60}")
    print("MORPHOLOGY STUDY SUMMARY (config.txt params)")
    print(f"{'='*60}")
    print(f"{'#':>5}  {'Loss':>10}  {'Intf':>8}  {'Donor':>7}  "
          f"{'Domains':>8}  {'Jsc':>8}")
    print("-"*55)
    for r in sorted(results, key=lambda x: x['loss']):
        print(f"{r['idx']:>5}  {r['loss']:>10.3e}  "
              f"{r['metrics']['interface_density']:>8.4f}  "
              f"{r['metrics']['donor_frac']:>7.3f}  "
              f"{r['metrics']['n_donor_domains']:>8}  "
              f"{r['jsc']:>8.4f}")

    loss_vals = [r['loss'] for r in results]
    print(f"\nLoss range: [{min(loss_vals):.3e}, {max(loss_vals):.3e}]")
    print(f"Loss mean : {np.mean(loss_vals):.3e}")

    return results


# =============================================================================
# SECTION 15 — MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', '--data-dir', dest='data_dir', default='.')
    ap.add_argument('--save_dir', '--save-dir', dest='save_dir', default='.')
    ap.add_argument('--n_morphs', '--n-morphs', dest='n_morphs', type=int, default=8,
                    help='Number of morphologies to study')
    ap.add_argument('--spread',         default='jsc',
                    choices=['uniform','jsc','random'],
                    help='How to spread morphology selection across dataset')
    ap.add_argument('--morph_indices', '--morph-indices', dest='morph_indices', default=None,
                    help='Comma-separated list of specific indices, e.g. 0,100,500')
    ap.add_argument('--n_epochs', '--n-epochs', dest='n_epochs', type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*60)
    print("OPV Morphology Study — Fixed Config.txt Parameters")
    print("="*60)
    print("Only morphology structure varies. Material params are FIXED.")
    print()

    # Fixed config params — same for every morphology
    cfg  = ConfigParams()
    norm = NormParams(cfg)

    # Load data
    morphs, params, jsc_gt = load_data(args.data_dir)

    # Select morphologies
    if args.morph_indices:
        indices = [int(x.strip()) for x in args.morph_indices.split(',')]
        print(f"Using specified indices: {indices}")
    else:
        indices = select_morphologies(jsc_gt, args.n_morphs,
                                       spread=args.spread)

    # Run study
    results = run_study(morphs, jsc_gt, indices, norm,
                        args.n_epochs, args.save_dir)

    # Summary plots
    plot_summary(results, args.save_dir)
    if len(results) >= 3:
        plot_correlation_matrix(results, args.save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()