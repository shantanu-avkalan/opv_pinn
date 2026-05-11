"""
OPV Pure-Physics PINN — v3 (all normalisation issues fixed)
============================================================
Fixes from v2:
  1. k_diss: Onsager-Braun formula had missing q^2 and gives near-zero
             under dark conditions. Use field-assisted k_diss=1e7 s-1
             which is physically correct for OPV under illumination.
  2. n0: reduced from 1e22 to 1e21 m^-3 so krec_nd ~ 3.5 (was 35).
  3. k_diss formula in residuals: D = k_diss * X * |E|
             (dissociation driven by local field, not field-independent)

All dimensionless params now in [1e-3, 1e3]:
  debye=0.048, Gx_nd=19.3, kx_nd=1.93, kdiss_nd=19.3, krec_nd=3.55

Expected Jsc: 0.3 - 3 mA/cm2 (J_nd ~ 0.5-3, J_scale=0.83 mA/cm2)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# 1.  PHYSICAL PARAMETERS
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
    E_g     = 1.1         # eV
    a       = 1.8e-9      # m
    tau_x   = 1e-6        # s
    Gx      = 1e28        # m^-3 s^-1
    Height  = 100e-9      # m

    def __init__(self):
        self.VT      = self.kb * self.T / self.q
        self.Dn      = self.mu_n * self.VT
        self.Dp      = self.mu_p * self.VT
        self.Dx      = self.mu_x * self.VT
        self.kx      = 1.0 / self.tau_x
        self.eps_avg = (self.eps_D + self.eps_A) / 2
        self.eps_si  = self.eps_avg * self.eps0

        # FIX 1: Corrected Onsager-Braun (has q^2, not q)
        self.E_bind  = self.q**2 / (4*np.pi*self.eps_si*self.a)   # Joules
        E_bind_eV    = self.E_bind / self.q
        # Dark OB rate — very small for low-eps organics
        k_diss_dark  = self.kx * np.exp(-self.E_bind / (self.kb*self.T))

        # Field-assisted OB rate at V_bi/L0 field
        V_bi   = self.E_g - 2*self.VT
        E_f    = V_bi / self.Height             # ~1.05e7 V/m
        b_OB   = self.q**3 * E_f / (8*np.pi*self.eps_si*(self.kb*self.T)**2)
        k_diss_field = k_diss_dark * (1 + b_OB + b_OB**2/6)

        # Use physically motivated value: 1e7 s^-1
        # (field-assisted dissociation, consistent with OPV literature)
        self.k_diss  = 1e7

        # Langevin recombination
        self.k_rec   = self.q*(self.mu_n+self.mu_p)/self.eps_si

        self.V_bi    = self.E_g - 2*self.VT

        print(f"\nPhysical params:")
        print(f"  VT       = {self.VT:.5f} V")
        print(f"  Dn       = {self.Dn:.4e} m2/s")
        print(f"  E_bind   = {E_bind_eV:.3f} eV")
        print(f"  k_diss   = {self.k_diss:.3e} s-1  (field-assisted)")
        print(f"  k_rec    = {self.k_rec:.3e} m3/s")
        print(f"  V_bi     = {self.V_bi:.4f} V")


# =============================================================================
# 2.  NON-DIMENSIONALISATION
# =============================================================================
class NormParams:
    def __init__(self, p: PhysicalParams):
        self.L0   = p.Height
        self.V0   = p.E_g            # 1.1 V — keeps phi ~ O(1)
        self.n0   = 1e21             # FIX 2: reduced from 1e22 → krec_nd ~3.5
        self.tau0 = self.L0**2 / p.Dn

        self.mu_n_nd  = 1.0
        self.mu_p_nd  = p.mu_p / p.mu_n
        self.mu_x_nd  = p.Dx   / p.Dn

        self.Gx_nd    = p.Gx    * self.tau0 / self.n0
        self.kx_nd    = p.kx    * self.tau0
        self.kdiss_nd = p.k_diss * self.tau0       # FIX 1: now non-zero
        self.krec_nd  = p.k_rec * self.n0 * self.tau0

        self.debye_ratio = p.q * self.n0 * self.L0**2 / (p.eps_si * self.V0)
        self.V_bi_nd     = p.V_bi / self.V0

        # Current scale: J_phys (A/m2) = J_nd * J_scale
        self.J_scale  = self.n0 * p.Dn * p.q / self.L0  # A/m2

        self._sanity()

    def _sanity(self):
        items = {
            "debye_ratio": self.debye_ratio,
            "Gx_nd"      : self.Gx_nd,
            "kx_nd"      : self.kx_nd,
            "kdiss_nd"   : self.kdiss_nd,
            "krec_nd"    : self.krec_nd,
            "mu_p_nd"    : self.mu_p_nd,
            "V_bi_nd"    : self.V_bi_nd,
        }
        print("\nSanity check:")
        ok_all = True
        for name, val in items.items():
            ok = 1e-3 <= abs(val) <= 1e3
            if not ok: ok_all = False
            print(f"  {name:14s} = {val:10.4f}  [{'OK' if ok else 'WARN'}]")
        print(f"  J_scale      = {self.J_scale*0.1:.4f} mA/cm2 per J_nd")
        print("  All params OK\n" if ok_all else "  SOME PARAMS OUT OF RANGE\n")


# =============================================================================
# 3.  ARCHITECTURE
# =============================================================================
class FieldNet(nn.Module):
    def __init__(self, n_hidden=5, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class OPV_PINN(nn.Module):
    def __init__(self, norm: NormParams, V_app_nd: float = 0.0):
        super().__init__()
        self.norm    = norm
        self.V_app   = V_app_nd
        self.net_phi = FieldNet()
        self.net_n   = FieldNet()
        self.net_p   = FieldNet()
        self.net_X   = FieldNet()

    def forward(self, x):
        # phi: built-in linear profile + network correction
        phi_0 =  self.norm.V_bi_nd / 2
        phi_1 = -self.norm.V_bi_nd / 2 + self.V_app
        phi   = phi_0*(1-x) + phi_1*x + x*(1-x)*self.net_phi(x)

        # n: 1 at anode → 0 at cathode
        n = (1-x) + x*(1-x)*torch.nn.functional.softplus(self.net_n(x))

        # p: 0 at anode → 1 at cathode
        p = x + x*(1-x)*torch.nn.functional.softplus(self.net_p(x))

        # X: zero at contacts, positive in bulk
        X = x*(1-x)*torch.nn.functional.softplus(self.net_X(x) + 1.0)

        return phi, n, p, X


# =============================================================================
# 4.  PHYSICS RESIDUALS
# =============================================================================
def _d1(f, x):
    return torch.autograd.grad(
        f, x, torch.ones_like(f), create_graph=True, retain_graph=True)[0]

def compute_residuals(model, x, norm):
    x   = x.requires_grad_(True)
    phi, n, p, X = model(x)

    dphi = _d1(phi, x)
    dn   = _d1(n,   x)
    dp   = _d1(p,   x)
    dX   = _d1(X,   x)
    d2phi = _d1(dphi, x)
    d2X   = _d1(dX,   x)

    E_mag = torch.abs(dphi)                          # |E| = |−dφ/dx|

    # FIX 3: dissociation = k_diss * X * |E|  (field-driven)
    D_diss = norm.kdiss_nd * X * E_mag
    R_rec  = norm.krec_nd  * n * p

    # Currents (dimensionless drift + diffusion)
    Jn = norm.mu_n_nd * (n * dphi + dn)
    Jp = norm.mu_p_nd * (p * dphi - dp)

    dJn = _d1(Jn, x)
    dJp = _d1(Jp, x)

    # PDE residuals
    res_P = d2phi  - norm.debye_ratio * (n - p)
    res_n = dJn    - R_rec + D_diss
    res_p = -dJp   - R_rec + D_diss
    res_X = norm.mu_x_nd * d2X - norm.kx_nd * X - D_diss + norm.Gx_nd

    return res_P, res_n, res_p, res_X, Jn, Jp


# =============================================================================
# 5.  LOSS
# =============================================================================
PHASE_W = [
    {'poisson': 20.0, 'n': 0.01, 'p': 0.01, 'X': 0.01, 'Jcons':   0.1},
    {'poisson':  5.0, 'n':  1.0, 'p':  1.0, 'X':  0.1, 'Jcons':  10.0},
    {'poisson':  1.0, 'n':  1.0, 'p':  1.0, 'X':  1.0, 'Jcons': 100.0},
]

def compute_loss(model, x_col, norm, weights):
    res_P, res_n, res_p, res_X, Jn, Jp = compute_residuals(model, x_col, norm)

    lP  = weights['poisson'] * (res_P**2).mean()
    ln  = weights['n']       * (res_n**2).mean()
    lp  = weights['p']       * (res_p**2).mean()
    lX  = weights['X']       * (res_X**2).mean()
    lJc = weights['Jcons']   * (Jn + Jp).var()

    total  = lP + ln + lp + lX + lJc
    losses = dict(total=total.item(), poisson=lP.item(),
                  n=ln.item(), p=lp.item(), X=lX.item(), Jcons=lJc.item())
    return total, losses


# =============================================================================
# 6.  TRAINING
# =============================================================================
def train(model, norm, n_epochs=15000, n_col=400, lr=5e-4, print_every=1000):
    x_u   = torch.linspace(0.01, 0.99, n_col).unsqueeze(1).to(DEVICE)
    x_i   = (torch.randn(n_col//4, 1)*0.04 + 0.5).clamp(0.01,0.99).to(DEVICE)
    x_col = torch.cat([x_u, x_i], dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3000, T_mult=2)

    history = {k: [] for k in ['total','poisson','n','p','X','Jcons']}

    def get_w(ep):
        f = ep / n_epochs
        return PHASE_W[0] if f < 0.20 else PHASE_W[1] if f < 0.50 else PHASE_W[2]

    print(f"Training {n_epochs} epochs | {len(x_col)} pts")
    print(f"{'Ep':>7}  {'Total':>9}  {'Poisson':>9}  {'n':>9}  {'p':>9}  {'X':>9}  {'Jcons':>9}")
    print("-"*68)

    for ep in range(n_epochs):
        opt.zero_grad()
        loss, losses = compute_loss(model, x_col, norm, get_w(ep))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        for k, v in losses.items(): history[k].append(v)
        if ep % print_every == 0 or ep == n_epochs-1:
            print(f"{ep:>7}  {losses['total']:>9.3e}  {losses['poisson']:>9.3e}  "
                  f"{losses['n']:>9.3e}  {losses['p']:>9.3e}  "
                  f"{losses['X']:>9.3e}  {losses['Jcons']:>9.3e}")
    return history


# =============================================================================
# 7.  JSC AND VALIDATION
# =============================================================================
def compute_jsc(model, norm, n_pts=1000):
    model.eval()
    x = torch.linspace(0.01, 0.99, n_pts).unsqueeze(1).to(DEVICE).requires_grad_(True)
    phi, n, p, _ = model(x)
    dphi = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=False)[0]
    dn   = torch.autograd.grad(n,   x, torch.ones_like(n),   create_graph=False)[0]
    dp   = torch.autograd.grad(p,   x, torch.ones_like(p),   create_graph=False)[0]

    Jn = norm.mu_n_nd * (n * dphi + dn)
    Jp = norm.mu_p_nd * (p * dphi - dp)
    Jt = (Jn + Jp).detach().cpu().numpy().flatten()
    Jn_np = Jn.detach().cpu().numpy().flatten()
    Jp_np = Jp.detach().cpu().numpy().flatten()

    # Dimensionalise: J_nd * J_scale * 0.1 → mA/cm2
    Jsc = float(np.mean(np.abs(Jt))) * norm.J_scale * 0.1
    return Jsc, Jn_np, Jp_np, Jt


def validate(model, norm, phys):
    model.eval()
    x_t = torch.linspace(0.01, 0.99, 500).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        phi, n, p, X = model(x_t)
    phi = phi.cpu().numpy().flatten()
    n   = n.cpu().numpy().flatten()
    p   = p.cpu().numpy().flatten()
    X   = X.cpu().numpy().flatten()

    Jsc, Jn, Jp, Jt = compute_jsc(model, norm, n_pts=500)
    flatness = np.std(Jt) / (np.mean(np.abs(Jt)) + 1e-12)

    print(f"\n{'='*60}")
    print(f"Jsc = {Jsc:.4f} mA/cm2")
    print(f"{'='*60}")
    print(f"\nCurrent conservation: std/mean = {flatness:.4f}  {'GOOD' if flatness<0.05 else 'NEEDS MORE TRAINING'}")
    print(f"\nField ranges (dimensionless):")
    print(f"  phi: [{phi.min():.3f}, {phi.max():.3f}]  expected ~[{-norm.V_bi_nd/2:.2f}, {norm.V_bi_nd/2:.2f}]")
    print(f"  n  : [{n.min():.4f}, {n.max():.4f}]")
    print(f"  p  : [{p.min():.4f}, {p.max():.4f}]")
    print(f"  X  : [{X.min():.4f}, {X.max():.4f}]")
    print(f"\nField ranges (physical):")
    print(f"  phi: [{phi.min()*norm.V0:.3f}, {phi.max()*norm.V0:.3f}] V  "
          f"expected ~[{-phys.V_bi/2:.3f}, {phys.V_bi/2:.3f}] V")
    print(f"  n  : n×n0 ~ [{n.min()*norm.n0:.2e}, {n.max()*norm.n0:.2e}] m^-3")

    return phi, n, p, X, Jn, Jp, Jt


# =============================================================================
# 8.  PLOTTING
# =============================================================================
def plot_results(model, norm, phys, history, phi, n, p, X, Jn, Jp, Jt):
    x_nm = np.linspace(1, 99, 500)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('OPV PINN — field solutions (v3: correct k_diss, n0, normalisation)',
                 fontsize=11, fontweight='bold')

    axes[0,0].plot(x_nm, phi*norm.V0, color='#3B8BD4', lw=2)
    axes[0,0].set(xlabel='Position (nm)', ylabel='φ (V)', title='Electric potential φ')
    axes[0,0].axvline(50, color='gray', lw=0.8, ls='--', alpha=0.5)

    axes[0,1].plot(x_nm, n, color='#1D9E75', lw=2, label='n (electrons)')
    axes[0,1].plot(x_nm, p, color='#E85D24', lw=2, label='p (holes)')
    axes[0,1].set(xlabel='Position (nm)', ylabel='Density (norm.)', title='Carrier densities')
    axes[0,1].legend(fontsize=9)
    axes[0,1].axvline(50, color='gray', lw=0.8, ls='--', alpha=0.5)

    axes[0,2].plot(x_nm, X, color='#BA7517', lw=2)
    axes[0,2].set(xlabel='Position (nm)', ylabel='X (norm.)', title='Exciton density X')
    axes[0,2].axvline(50, color='gray', lw=0.8, ls='--', alpha=0.5)

    axes[1,0].plot(x_nm, Jn, color='#1D9E75', lw=2, label='Jn')
    axes[1,0].plot(x_nm, Jp, color='#E85D24', lw=2, label='Jp')
    axes[1,0].plot(x_nm, Jt, color='#534AB7', lw=2, ls='--', label='J total')
    axes[1,0].set(xlabel='Position (nm)', ylabel='J (norm.)', title='Current density (J total = flat ✓)')
    axes[1,0].legend(fontsize=9)
    axes[1,0].axvline(50, color='gray', lw=0.8, ls='--', alpha=0.5)

    for k, c, lbl in [('total','black','total'),('poisson','#3B8BD4','Poisson'),
                       ('n','#1D9E75','n'),('p','#E85D24','p'),
                       ('X','#BA7517','X'),('Jcons','#534AB7','Jcons')]:
        axes[1,1].semilogy(history[k], color=c, lw=1.2 if k=='total' else 1.0, label=lbl)
    axes[1,1].set(xlabel='Epoch', ylabel='Loss (log)', title='Training loss')
    axes[1,1].legend(fontsize=8)

    E_phys = -np.gradient(phi*norm.V0, x_nm*1e-9) / 1e6  # MV/m
    axes[1,2].plot(x_nm, E_phys, color='#D85A30', lw=2)
    axes[1,2].set(xlabel='Position (nm)', ylabel='E (MV/m)', title='Electric field')
    axes[1,2].axvline(50, color='gray', lw=0.8, ls='--', alpha=0.5)

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('opv_pinn_v3.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 9.  J-V SWEEP
# =============================================================================
def jv_sweep(norm, phys, n_V=10, n_epochs_each=5000):
    V_phys = np.linspace(0, phys.V_bi*0.95, n_V)
    V_nd   = V_phys / norm.V0
    J_out  = []

    print(f"\nJ-V sweep ({n_V} points × {n_epochs_each} epochs each):")
    print(f"  {'V (V)':>8}  {'J (mA/cm2)':>12}")
    print("  " + "-"*22)

    for V in V_nd:
        m = OPV_PINN(norm, V_app_nd=float(V)).to(DEVICE)
        o = torch.optim.Adam(m.parameters(), lr=5e-4)
        x = torch.linspace(0.01, 0.99, 300).unsqueeze(1).to(DEVICE)
        for ep in range(n_epochs_each):
            o.zero_grad()
            loss, _ = compute_loss(m, x, norm, PHASE_W[2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()
        J, _, _, _ = compute_jsc(m, norm)
        J_out.append(J)
        print(f"  {V*norm.V0:>8.3f}  {J:>12.4f}")

    V_phys_arr = V_phys
    J_arr = np.array(J_out)

    # Metrics
    Jsc  = J_arr[0]
    Voc  = float(np.interp(0, -J_arr[::-1], V_phys_arr[::-1])) if J_arr[-1] < 0 else V_phys_arr[-1]
    P    = V_phys_arr * J_arr
    Pmax = P.max()
    FF   = Pmax / (Voc * Jsc) if Voc > 0 and Jsc > 0 else 0

    print(f"\nJsc  = {Jsc:.4f} mA/cm2")
    print(f"Voc  = {Voc:.4f} V")
    print(f"Pmax = {Pmax:.4f} mW/cm2")
    print(f"FF   = {FF:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(V_phys_arr, J_arr, 'o-', color='#3B8BD4', lw=2, markersize=6)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set(xlabel='Applied voltage (V)', ylabel='J (mA/cm2)', title='J-V curve')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('opv_jv_v3.png', dpi=150, bbox_inches='tight')
    plt.show()

    return V_phys_arr, J_arr


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    print("="*60)
    print("OPV Pure-Physics PINN  v3")
    print("="*60)

    phys  = PhysicalParams()
    norm  = NormParams(phys)
    model = OPV_PINN(norm, V_app_nd=0.0).to(DEVICE)
    print(f"Network params: {sum(p.numel() for p in model.parameters()):,}")

    history = train(model, norm, n_epochs=15000, n_col=400,
                    lr=5e-4, print_every=1000)

    phi, n, p, X, Jn, Jp, Jt = validate(model, norm, phys)
    plot_results(model, norm, phys, history, phi, n, p, X, Jn, Jp, Jt)

    run_jv = input("\nRun J-V sweep? [y/N]: ").strip().lower()
    if run_jv == 'y':
        jv_sweep(norm, phys)

    print("\nDone.")

if __name__ == "__main__":
    main()