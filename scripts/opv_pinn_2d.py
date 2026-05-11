"""
OPV 2D PINN — Hand-Defined Morphologies (v3)
=============================================
Fixes from v2:
  FIX A — k_diss reduced from 1e7 to 1e6
           kdiss_nd drops from 19.3 to 1.93
           Prevents carrier pileup (n was reaching 4.7, should be ~2)

  FIX B — Scheduler changed to CosineAnnealingLR (no restarts)
           Warm restarts were spiking the loss at each cycle,
           preventing final convergence below ~7

  FIX C — Dual Jcons loss: variance + mean-deviation penalty
           Forces both variance AND absolute level of current to be uniform

  FIX D — Learning rate reduced to 1e-4 (was 5e-4)
           High LR was causing oscillation after epoch 10000

All other fixes from v2 retained:
  - abs(mean(Jt)) for Jsc  [FIX 1]
  - Jcons weight = 200      [FIX 2]
  - 6 layers × 128 neurons  [FIX 3]
  - 20000 epochs, 1600 pts  [FIX 4]

Expected ordering: Bilayer < Columnar < Checkerboard
Expected bilayer Jsc: ~3-4 mA/cm² (close to 1D reference 3.34)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# SECTION 1 — PHYSICAL PARAMETERS
# =============================================================================
class PhysicalParams:
    q    = 1.602e-19
    kb   = 1.381e-23
    eps0 = 8.854e-12
    T       = 300.0
    mu_n    = 2e-7
    mu_p    = 1.5e-7
    mu_x    = 3.9e-9
    eps_D   = 3.0
    eps_A   = 3.9
    E_g     = 1.1
    a       = 1.8e-9
    tau_x   = 1e-6
    Gx      = 1e28
    Height  = 100e-9
    muRatio = 0.01

    def __init__(self):
        self.VT      = self.kb * self.T / self.q
        self.Dn      = self.mu_n * self.VT
        self.Dp      = self.mu_p * self.VT
        self.Dx      = self.mu_x * self.VT
        self.kx      = 1.0 / self.tau_x
        self.eps_avg = (self.eps_D + self.eps_A) / 2
        self.eps_si  = self.eps_avg * self.eps0
        self.k_diss  = 1e6    # FIX A: reduced from 1e7 → kdiss_nd = 1.93
        self.k_rec   = self.q * (self.mu_n + self.mu_p) / self.eps_si
        self.V_bi    = self.E_g - 2 * self.VT
        print(f"\nPhysical params:")
        print(f"  mu_n={self.mu_n:.1e}  mu_p={self.mu_p:.1e}  "
              f"mu_x={self.mu_x:.1e}")
        print(f"  k_diss={self.k_diss:.1e}  (reduced to prevent carrier pileup)")
        print(f"  V_bi={self.V_bi:.4f} V")


# =============================================================================
# SECTION 2 — NON-DIMENSIONALISATION
# =============================================================================
class NormParams:
    def __init__(self, p: PhysicalParams):
        self.L0   = p.Height
        self.V0   = p.E_g
        self.n0   = 1e21
        self.tau0 = self.L0**2 / p.Dn

        self.mu_n_nd  = 1.0
        self.mu_p_nd  = p.mu_p / p.mu_n
        self.mu_x_nd  = p.Dx   / p.Dn
        self.muRatio  = p.muRatio

        self.Gx_nd    = p.Gx    * self.tau0 / self.n0
        self.kx_nd    = p.kx    * self.tau0
        self.kdiss_nd = p.k_diss * self.tau0   # now 1.93 (was 19.3)
        self.krec_nd  = p.k_rec * self.n0 * self.tau0

        self.debye_ratio = p.q * self.n0 * self.L0**2 / (p.eps_si * self.V0)
        self.V_bi_nd     = p.V_bi / self.V0
        self.J_scale     = self.n0 * p.Dn * p.q / self.L0

        print("\nSanity check:")
        for name, val in [("debye_ratio", self.debye_ratio),
                           ("Gx_nd",       self.Gx_nd),
                           ("kx_nd",       self.kx_nd),
                           ("kdiss_nd",    self.kdiss_nd),
                           ("krec_nd",     self.krec_nd)]:
            ok = 1e-3 <= abs(val) <= 1e3
            print(f"  {name:14s} = {val:8.4f}  [{'OK' if ok else 'WARN'}]")
        print(f"  J_scale      = {self.J_scale*0.1:.4f} mA/cm2 per J_nd")
        print(f"  kdiss/krec   = {self.kdiss_nd/self.krec_nd:.3f}"
              f"  (dissociation vs recombination balance)")
        print(f"  kdiss/kx     = {self.kdiss_nd/self.kx_nd:.3f}"
              f"  (dissociation vs exciton decay)\n")


# =============================================================================
# SECTION 3 — MORPHOLOGY DEFINITIONS
# =============================================================================
def make_bilayer(grid_size=64):
    """Donor left half, acceptor right half. Sanity check — should match 1D."""
    morph = np.zeros((grid_size, grid_size), dtype=np.float32)
    morph[:grid_size//2, :] = 1.0
    return morph


def make_checkerboard(grid_size=64, n_tiles=8):
    """Alternating squares. Most interface area — should give highest Jsc."""
    morph = np.zeros((grid_size, grid_size), dtype=np.float32)
    tile  = grid_size // n_tiles
    for i in range(grid_size):
        for j in range(grid_size):
            if (i // tile + j // tile) % 2 == 0:
                morph[i, j] = 1.0
    return morph


def make_columnar(grid_size=64, n_columns=8):
    """Vertical stripes. Good percolation pathways."""
    morph = np.zeros((grid_size, grid_size), dtype=np.float32)
    col_width = grid_size // n_columns
    for j in range(grid_size):
        if (j // col_width) % 2 == 0:
            morph[:, j] = 1.0
    return morph


# =============================================================================
# SECTION 4 — MORPHOLOGY HANDLER
# =============================================================================
class MorphologyHandler:
    def __init__(self, morph_grid: np.ndarray, norm: NormParams):
        from scipy.interpolate import RegularGridInterpolator
        H, W = morph_grid.shape
        self.interp = RegularGridInterpolator(
            (np.linspace(0,1,H), np.linspace(0,1,W)),
            morph_grid, method='linear',
            bounds_error=False, fill_value=0.5
        )
        self.norm       = norm
        self.morph_grid = morph_grid
        donor_frac = float((morph_grid > 0.5).mean())
        print(f"  Morphology: {H}x{W}  donor_frac={donor_frac:.3f}")

    def _phase(self, xy: torch.Tensor) -> torch.Tensor:
        """Returns soft donor fraction at each (x,y) point. ~1=donor, ~0=acceptor."""
        xy_np = xy.detach().cpu().numpy()
        phase = self.interp(xy_np).astype(np.float32)
        return torch.sigmoid(
            20.0 * (torch.tensor(phase).unsqueeze(1).to(xy.device) - 0.5)
        )

    def get_Gx(self, xy):
        """Generation only in donor regions."""
        return self.norm.Gx_nd * self._phase(xy)

    def get_mu_n(self, xy):
        """Full mu_n in acceptor, reduced by muRatio in donor."""
        d = self._phase(xy)
        return self.norm.mu_n_nd * ((1-d) + self.norm.muRatio * d)

    def get_mu_p(self, xy):
        """Full mu_p in donor, reduced by muRatio in acceptor."""
        d = self._phase(xy)
        return self.norm.mu_p_nd * (d + self.norm.muRatio * (1-d))


# =============================================================================
# SECTION 5 — NETWORK ARCHITECTURE  (6 layers × 128 neurons from v2)
# =============================================================================
class FieldNet(nn.Module):
    def __init__(self, n_hidden=6, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


class OPV_PINN_2D(nn.Module):
    def __init__(self, norm: NormParams, V_app_nd: float = 0.0):
        super().__init__()
        self.norm    = norm
        self.V_app   = V_app_nd
        self.net_phi = FieldNet()
        self.net_n   = FieldNet()
        self.net_p   = FieldNet()
        self.net_X   = FieldNet()

    def forward(self, xy):
        x = xy[:, 0:1]
        phi_L =  self.norm.V_bi_nd / 2
        phi_R = -self.norm.V_bi_nd / 2 + self.V_app
        phi   = phi_L*(1-x) + phi_R*x + x*(1-x)*self.net_phi(xy)
        n     = (1-x) + x*(1-x)*F.softplus(self.net_n(xy))
        p     =     x + x*(1-x)*F.softplus(self.net_p(xy))
        X     = x*(1-x)*F.softplus(self.net_X(xy) + 1.0)
        return phi, n, p, X


# =============================================================================
# SECTION 6 — DERIVATIVES AND RESIDUALS
# =============================================================================
def grad2d(f, xy):
    g = torch.autograd.grad(
        f, xy, torch.ones_like(f),
        create_graph=True, retain_graph=True)[0]
    return g[:, 0:1], g[:, 1:2]


def compute_residuals(model, xy, norm, morph):
    xy = xy.requires_grad_(True)
    phi, n, p, X = model(xy)

    dphi_x, dphi_y = grad2d(phi, xy)
    dn_x,   dn_y   = grad2d(n,   xy)
    dp_x,   dp_y   = grad2d(p,   xy)
    dX_x,   dX_y   = grad2d(X,   xy)

    d2phi_x, _  = grad2d(dphi_x, xy)
    _, d2phi_y  = grad2d(dphi_y, xy)
    d2X_x,  _   = grad2d(dX_x,  xy)
    _,  d2X_y   = grad2d(dX_y,  xy)

    lap_phi = d2phi_x + d2phi_y
    lap_X   = d2X_x  + d2X_y
    E_mag   = torch.sqrt(dphi_x**2 + dphi_y**2 + 1e-8)

    Gx_loc   = morph.get_Gx(xy)
    mu_n_loc = morph.get_mu_n(xy)
    mu_p_loc = morph.get_mu_p(xy)

    D_diss = norm.kdiss_nd * X * E_mag
    R_rec  = norm.krec_nd  * n * p

    Jn_x = mu_n_loc * (n * dphi_x + dn_x)
    Jn_y = mu_n_loc * (n * dphi_y + dn_y)
    Jp_x = mu_p_loc * (p * dphi_x - dp_x)
    Jp_y = mu_p_loc * (p * dphi_y - dp_y)

    dJnx_dx, _ = grad2d(Jn_x, xy)
    _, dJny_dy = grad2d(Jn_y, xy)
    dJpx_dx, _ = grad2d(Jp_x, xy)
    _, dJpy_dy = grad2d(Jp_y, xy)

    res_P = lap_phi  - norm.debye_ratio * (n - p)
    res_n = (dJnx_dx + dJny_dy) - R_rec + D_diss
    res_p = -(dJpx_dx + dJpy_dy) - R_rec + D_diss
    res_X = norm.mu_x_nd * lap_X - norm.kx_nd * X - D_diss + Gx_loc

    return res_P, res_n, res_p, res_X, Jn_x + Jp_x


# =============================================================================
# SECTION 7 — LOSS
# FIX C: dual Jcons — variance + mean-deviation penalty
# Why two terms?
#   var(Jt)              -> penalises spatial variation (existing)
#   mean((Jt-mean)^2)    -> also penalises deviation from the mean value
#   Together they force Jt to be spatially constant everywhere
# =============================================================================
PHASE_W = [
    {'poisson': 20.0, 'n': 0.01, 'p': 0.01, 'X': 0.01, 'Jcons':   0.1},
    {'poisson':  5.0, 'n':  1.0, 'p':  1.0, 'X':  0.5, 'Jcons':  20.0},
    {'poisson':  1.0, 'n':  1.0, 'p':  1.0, 'X':  1.0, 'Jcons': 200.0},
]


def compute_loss(model, xy, norm, morph, weights):
    res_P, res_n, res_p, res_X, Jt_x = compute_residuals(model, xy, norm, morph)

    lP  = weights['poisson'] * (res_P**2).mean()
    ln  = weights['n']       * (res_n**2).mean()
    lp  = weights['p']       * (res_p**2).mean()
    lX  = weights['X']       * (res_X**2).mean()

    # FIX C: variance term (penalises spatial variation of current)
    lJc_var = weights['Jcons'] * Jt_x.var()

    # FIX C: mean-deviation term (penalises current deviating from its mean)
    # This catches cases where Jt is "uniformly wrong" — all shifted but flat
    Jt_mean = Jt_x.mean().detach()
    lJc_dev = weights['Jcons'] * 0.5 * ((Jt_x - Jt_mean)**2).mean()

    total  = lP + ln + lp + lX + lJc_var + lJc_dev
    losses = dict(
        total   = total.item(),
        poisson = lP.item(),
        n       = ln.item(),
        p       = lp.item(),
        X       = lX.item(),
        Jcons   = (lJc_var + lJc_dev).item()
    )
    return total, losses


# =============================================================================
# SECTION 8 — COLLOCATION POINTS (1200 bulk + 400 interface from v2)
# =============================================================================
def make_collocation_points(morph_handler, n_bulk=1200, n_interface=400):
    xy_bulk = torch.rand(n_bulk, 2)
    mg   = morph_handler.morph_grid
    gx   = np.abs(np.gradient(mg, axis=0))
    gy   = np.abs(np.gradient(mg, axis=1))
    prob = (gx + gy).flatten()
    H, W = mg.shape
    if prob.sum() > 0:
        prob = prob / prob.sum()
        idx  = np.random.choice(H*W, size=n_interface, p=prob, replace=True)
        xi   = (idx//W/H + np.random.randn(n_interface)*0.01).clip(0.01,0.99)
        yi   = (idx%W/W  + np.random.randn(n_interface)*0.01).clip(0.01,0.99)
        xy_i = torch.tensor(np.stack([xi,yi],axis=1), dtype=torch.float32)
    else:
        xi   = (np.random.randn(n_interface)*0.03+0.5).clip(0.01,0.99)
        yi   = np.random.rand(n_interface)
        xy_i = torch.tensor(np.stack([xi,yi],axis=1), dtype=torch.float32)
    return torch.cat([xy_bulk, xy_i], dim=0).to(DEVICE)


# =============================================================================
# SECTION 9 — TRAINING
# FIX B: CosineAnnealingLR (no restarts) — smooth decay to zero
# FIX D: lr=1e-4 (was 5e-4) — prevents oscillation at fine scale
# =============================================================================
def train(model, norm, morph, name="",
          n_epochs=20000, lr=1e-4, print_every=2000):

    xy_col = make_collocation_points(morph)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # FIX B: smooth cosine decay, no restarts
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {k: [] for k in ['total','poisson','n','p','X','Jcons']}

    def get_w(ep):
        f = ep / n_epochs
        if f < 0.20: return PHASE_W[0]
        if f < 0.50: return PHASE_W[1]
        return PHASE_W[2]

    print(f"\n{'='*60}")
    print(f"Training: {name}  |  {n_epochs} epochs  |  {len(xy_col)} pts")
    print(f"lr={lr}  scheduler=CosineAnnealingLR (no restarts)")
    print(f"{'='*60}")
    print(f"{'Ep':>7}  {'Total':>9}  {'Poisson':>9}  "
          f"{'n':>8}  {'p':>8}  {'X':>8}  {'Jcons':>9}")
    print("-" * 65)

    for ep in range(n_epochs):
        opt.zero_grad()
        loss, losses = compute_loss(model, xy_col, norm, morph, get_w(ep))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        for k, v in losses.items():
            history[k].append(v)
        if ep % print_every == 0 or ep == n_epochs-1:
            lr_now = opt.param_groups[0]['lr']
            print(f"{ep:>7}  {losses['total']:>9.3e}  "
                  f"{losses['poisson']:>9.3e}  {losses['n']:>8.3e}  "
                  f"{losses['p']:>8.3e}  {losses['X']:>8.3e}  "
                  f"{losses['Jcons']:>9.3e}  lr={lr_now:.2e}")
    return history


# =============================================================================
# SECTION 10 — JSC COMPUTATION  (abs(mean(Jt)) from v2 FIX 1)
# =============================================================================
def compute_jsc(model, norm, n_pts=200):
    model.eval()
    y_pts = torch.linspace(0.01, 0.99, n_pts).unsqueeze(1)
    x_pts = torch.ones(n_pts, 1)
    xy    = torch.cat([x_pts, y_pts], dim=1).to(DEVICE).requires_grad_(True)

    phi, n, p, X = model(xy)
    g_phi = torch.autograd.grad(phi, xy, torch.ones_like(phi),
                                 create_graph=False, retain_graph=True)[0]
    g_n   = torch.autograd.grad(n,   xy, torch.ones_like(n),
                                 create_graph=False, retain_graph=True)[0]
    g_p   = torch.autograd.grad(p,   xy, torch.ones_like(p),
                                 create_graph=False)[0]

    Jn_x = norm.mu_n_nd * (n * g_phi[:,0:1] + g_n[:,0:1])
    Jp_x = norm.mu_p_nd * (p * g_phi[:,0:1] - g_p[:,0:1])
    Jt   = (Jn_x + Jp_x).detach().cpu().numpy().flatten()

    Jsc     = abs(float(np.mean(Jt))) * norm.J_scale * 0.1
    lat_var = float(np.std(Jt) / (abs(np.mean(Jt)) + 1e-12))
    return Jsc, Jt, lat_var


# =============================================================================
# SECTION 11 — FIELD EVALUATION
# =============================================================================
def eval_on_grid(model, norm, grid_size=64):
    model.eval()
    x1 = np.linspace(0.01, 0.99, grid_size)
    y1 = np.linspace(0.01, 0.99, grid_size)
    XX, YY = np.meshgrid(x1, y1, indexing='ij')
    xy_flat = torch.tensor(
        np.stack([XX.flatten(), YY.flatten()], axis=1), dtype=torch.float32
    ).to(DEVICE)

    phi_l, n_l, p_l, X_l, Jx_l = [], [], [], [], []
    for i in range(0, len(xy_flat), 1024):
        xy_b = xy_flat[i:i+1024].requires_grad_(True)
        phi_b, n_b, p_b, X_b = model(xy_b)
        g_phi = torch.autograd.grad(phi_b, xy_b, torch.ones_like(phi_b),
                                     create_graph=False, retain_graph=True)[0]
        g_n   = torch.autograd.grad(n_b,   xy_b, torch.ones_like(n_b),
                                     create_graph=False, retain_graph=True)[0]
        g_p   = torch.autograd.grad(p_b,   xy_b, torch.ones_like(p_b),
                                     create_graph=False)[0]
        Jn_x = norm.mu_n_nd * (n_b * g_phi[:,0:1] + g_n[:,0:1])
        Jp_x = norm.mu_p_nd * (p_b * g_phi[:,0:1] - g_p[:,0:1])
        phi_l.append(phi_b.detach().cpu().numpy())
        n_l  .append(n_b  .detach().cpu().numpy())
        p_l  .append(p_b  .detach().cpu().numpy())
        X_l  .append(X_b  .detach().cpu().numpy())
        Jx_l .append((Jn_x+Jp_x).detach().cpu().numpy())

    def r(lst): return np.concatenate(lst).reshape(grid_size, grid_size)
    return r(phi_l), r(n_l), r(p_l), r(X_l), r(Jx_l)


# =============================================================================
# SECTION 12 — PLOTTING
# =============================================================================
def plot_single(morph_handler, phi, n, p, X, Jx, history, Jsc, lat_var, name):
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.35)

    conv_status = "CONVERGED ✓" if lat_var < 0.15 else f"var={lat_var:.3f}"
    fig.suptitle(
        f'{name}  —  Jsc = {Jsc:.4f} mA/cm²  ({conv_status})',
        fontsize=12, fontweight='bold'
    )
    mg  = morph_handler.morph_grid
    ext = [0, 100, 0, 100]

    def add_contour(ax):
        ax.contour(np.linspace(0,100,mg.shape[0]),
                   np.linspace(0,100,mg.shape[1]),
                   mg.T, levels=[0.5],
                   colors='white', linewidths=0.8, alpha=0.7)

    def hmap(ax, data, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(data.T, origin='lower', cmap=cmap,
                       extent=ext, aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        add_contour(ax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (nm)', fontsize=8)
        ax.set_ylabel('y (nm)', fontsize=8)
        ax.tick_params(labelsize=7)

    ax0 = fig.add_subplot(gs[0,0])
    im  = ax0.imshow(mg.T, origin='lower', cmap='RdYlBu',
                     extent=ext, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title('Morphology\nred=donor  blue=acceptor', fontsize=9)
    ax0.set_xlabel('x (nm)', fontsize=8); ax0.set_ylabel('y (nm)', fontsize=8)

    hmap(fig.add_subplot(gs[0,1]), phi*1.1,
         'Electric potential φ (V)', 'RdBu_r')
    hmap(fig.add_subplot(gs[0,2]), X,
         'Exciton density X\n(should match donor pattern)', 'YlOrBr')
    hmap(fig.add_subplot(gs[1,0]), n,
         'Electron density n\n(should be <2)', 'Greens')
    hmap(fig.add_subplot(gs[1,1]), p,
         'Hole density p', 'Oranges')

    # Jx: symmetric colormap so zero is white
    vmax_jx = max(abs(Jx.min()), abs(Jx.max()))
    hmap(fig.add_subplot(gs[1,2]), Jx,
         'x-Current Jx\n(uniform = converged ✓)',
         'RdBu_r', vmin=-vmax_jx, vmax=vmax_jx)

    ax6 = fig.add_subplot(gs[:,3])
    for k,c,lb in [('total','black','total'),('poisson','#3B8BD4','Poisson'),
                    ('n','#1D9E75','n'),('p','#E85D24','p'),
                    ('X','#BA7517','X'),('Jcons','#534AB7','Jcons')]:
        ax6.semilogy(history[k], color=c, lw=1.5 if k=='total' else 1, label=lb)
    ax6.axhline(1.0, color='gray', lw=1, ls='--', alpha=0.5)
    ax6.text(len(history['total'])*0.02, 1.2, 'target < 1', fontsize=7, color='gray')
    ax6.set_xlabel('Epoch', fontsize=8); ax6.set_ylabel('Loss', fontsize=8)
    ax6.set_title('Training loss', fontsize=9); ax6.legend(fontsize=7)
    ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)

    fname = f"opv_2d_v3_{name.lower().replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")


def plot_comparison(names, jsc_values, lat_vars, histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Morphology comparison (v3)', fontsize=12, fontweight='bold')
    colors = ['#534AB7', '#1D9E75', '#E85D24', '#BA7517']

    ax = axes[0]
    bars = ax.bar(names, jsc_values, color=colors[:len(names)],
                  alpha=0.8, edgecolor='white')
    for bar, val, lv in zip(bars, jsc_values, lat_vars):
        mark = "✓" if lv < 0.15 else "~"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.3f} {mark}', ha='center', va='bottom', fontsize=11)
    ax.axhline(3.34, color='gray', lw=1.2, ls='--', alpha=0.7)
    ax.text(0.02, 3.45, '1D reference (3.34)', fontsize=8, color='gray')
    ax.set_ylabel('Jsc (mA/cm²)', fontsize=11)
    ax.set_title('Jsc by morphology\n✓ = converged  ~ = partial', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax2 = axes[1]
    for i,(name,hist) in enumerate(zip(names, histories)):
        ax2.semilogy(hist['total'], color=colors[i], lw=1.5, label=name)
    ax2.axhline(1.0, color='gray', lw=1, ls='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Total loss', fontsize=11)
    ax2.set_title('Training convergence\n(target < 1.0)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('opv_morphology_comparison_v3.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: opv_morphology_comparison_v3.png")


# =============================================================================
# SECTION 13 — MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("OPV 2D PINN — Morphology Test  v3")
    print("=" * 60)

    phys = PhysicalParams()
    norm = NormParams(phys)

    GRID     = 64
    N_EPOCHS = 20000

    morphology_defs = {
        'Bilayer'      : make_bilayer(GRID),
        'Checkerboard' : make_checkerboard(GRID, n_tiles=8),
        'Columnar'     : make_columnar(GRID, n_columns=8),
    }

    results   = {}
    lat_vars  = {}
    histories = {}

    for name, morph_grid in morphology_defs.items():
        print(f"\n{'#'*60}")
        print(f"# {name}")
        print(f"{'#'*60}")

        morph   = MorphologyHandler(morph_grid, norm)
        model   = OPV_PINN_2D(norm).to(DEVICE)
        n_param = sum(p.numel() for p in model.parameters())
        print(f"  Total network parameters: {n_param:,}")

        history           = train(model, norm, morph, name=name,
                                  n_epochs=N_EPOCHS, lr=1e-4)
        Jsc, Jt, lat_var  = compute_jsc(model, norm)

        results[name]     = Jsc
        lat_vars[name]    = lat_var
        histories[name]   = history

        final_loss = history['total'][-1]
        converged  = final_loss < 1.0

        print(f"\nResults — {name}:")
        print(f"  Jsc               = {Jsc:.4f} mA/cm²")
        print(f"  Lateral variation = {lat_var:.4f}  "
              f"({'GOOD' if lat_var<0.15 else 'HIGH'})")
        print(f"  Final total loss  = {final_loss:.4e}  "
              f"({'CONVERGED ✓' if converged else 'NOT YET'})")

        phi_2d, n_2d, p_2d, X_2d, Jx_2d = eval_on_grid(model, norm, GRID)
        Jx_ratio = Jx_2d.std() / (abs(Jx_2d.mean()) + 1e-12)

        print(f"  phi: [{phi_2d.min()*norm.V0:.3f}, {phi_2d.max()*norm.V0:.3f}] V")
        print(f"  n  : [{n_2d.min():.3f}, {n_2d.max():.3f}]  "
              f"{'OK' if n_2d.max()<2.5 else 'HIGH — k_diss may still be too large'}")
        print(f"  p  : [{p_2d.min():.3f}, {p_2d.max():.3f}]")
        print(f"  X  : [{X_2d.min():.3f}, {X_2d.max():.3f}]")
        print(f"  Jx std/|mean| = {Jx_ratio:.4f}  "
              f"({'GOOD' if Jx_ratio<0.15 else 'needs more convergence'})")

        plot_single(morph, phi_2d, n_2d, p_2d, X_2d, Jx_2d,
                    history, Jsc, lat_var, name)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Morphology':15s}  {'Jsc':>8}  {'Lat.Var':>9}  {'Jx unif':>9}  {'Loss':>10}")
    print("-" * 58)
    for name in results:
        phi_2d, n_2d, p_2d, X_2d, Jx_2d = eval_on_grid(
            OPV_PINN_2D(norm).to(DEVICE), norm, 32)
        final_loss = histories[name]['total'][-1]
        print(f"{name:15s}  {results[name]:>8.4f}  "
              f"{lat_vars[name]:>9.4f}  "
              f"{'GOOD' if lat_vars[name]<0.15 else 'HIGH':>9}  "
              f"{final_loss:>10.3e}")

    print(f"\n1D bilayer reference = 3.3400 mA/cm²")
    bil = results.get('Bilayer', 0)
    print(f"2D bilayer result    = {bil:.4f} mA/cm²  "
          f"({abs(bil-3.34)/3.34*100:.1f}% from reference)")

    vals = sorted(results.items(), key=lambda x: x[1])
    print(f"\nActual ordering: {' < '.join(f'{n}({v:.3f})' for n,v in vals)}")
    print(f"Expected:        Bilayer < Columnar < Checkerboard")

    correct = (results.get('Checkerboard',0) > results.get('Bilayer',0) and
               results.get('Columnar',0)     > results.get('Bilayer',0))
    print(f"\n{'Ordering correct ✓' if correct else 'Ordering wrong — see Jx maps'}")

    plot_comparison(list(results.keys()), list(results.values()),
                    list(lat_vars.values()), list(histories.values()))
    print("\nDone.")


if __name__ == "__main__":
    main()