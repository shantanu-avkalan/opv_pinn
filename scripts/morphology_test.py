"""
OPV 2D PINN v6 — dataset params validation, physics improvements
=================================================================
Changes from previous version:
  CHG 2 — Shared encoder architecture (one encoder, four output heads)
           Fewer params, more consistent field representations
  CHG 3 — Adaptive curriculum: phase advances when loss actually converges
           not at fixed epoch fractions
  CHG 4 — Interface-weighted loss: 4x weight at donor-acceptor boundaries
           Reduces sharp residual spikes at interfaces
  CHG 5 — Squared gradient sampling + n_intf=600
           More collocation points at sharpest interface regions

Run:
    python opv_pinn_2d.py --data_dir /path/to/data --morph_idx 0
    python opv_pinn_2d.py --data_dir /path/to/data --validate 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math, os, argparse

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# SECTION 1 — PHYSICAL PARAMETERS
# =============================================================================
class PhysicalParams:
    q=1.602e-19; kb=1.381e-23; eps0=8.854e-12
    T=300.0; eps_D=3.0; eps_A=3.9; E_g=1.1
    Height=100e-9; muRatio=0.01; Gx=1e28
    mu_n=2e-7; mu_p=1.5e-7; mu_x=3.9e-9
    tau_x=1e-6; kdiss_factor=1.0

    def __init__(self):
        self.VT      = self.kb*self.T/self.q
        self.eps_avg = (self.eps_D+self.eps_A)/2
        self.eps_si  = self.eps_avg*self.eps0
        self.V_bi    = self.E_g - 2*self.VT
        self._recompute()

    def _recompute(self):
        self.Dn     = self.mu_n*self.VT
        self.Dp     = self.mu_p*self.VT
        self.Dx     = self.mu_x*self.VT
        self.kx     = 1.0/self.tau_x
        self.k_diss = self.kdiss_factor*1e6
        self.k_rec  = 0.1*self.q*(self.mu_n+self.mu_p)/self.eps_si

    def update_from_dataset(self, row):
        self.mu_n=float(row[0]); self.mu_p=float(row[1]); self.mu_x=float(row[2])
        self.kdiss_factor=float(row[3]); self.tau_x=float(row[4])
        self._recompute()
        print(f"  Params: mu_n={self.mu_n:.1e} mu_p={self.mu_p:.1e} "
              f"mu_x={self.mu_x:.1e} tau_x={self.tau_x:.1e} kdf={self.kdiss_factor:.1f}")


# =============================================================================
# SECTION 2 — NON-DIMENSIONALISATION
# =============================================================================
class NormParams:
    def __init__(self, p: PhysicalParams):
        self.L0    = p.Height
        self.V0    = p.E_g
        self.VT_nd = p.VT/p.E_g

        _J_REF  = 3e21*(2e-7*p.VT)*p.q/p.Height
        self.n0 = _J_REF/(p.Dn*p.q/p.Height)
        self.tau0 = p.Height**2/p.Dn

        self.mu_n_nd = 1.0
        self.mu_p_nd = p.mu_p/p.mu_n
        self.mu_x_nd = p.Dx/p.Dn
        self.muRatio = p.muRatio

        self.Gx_nd    = p.Gx*self.tau0/self.n0
        self.kx_nd    = p.kx*self.tau0
        self.kdiss_nd = p.k_diss*self.tau0

        # Gx_scale clamped to [0,1] — never amplify generation
        self.Gx_scale = min(1.0, self.kx_nd/self.Gx_nd) if self.Gx_nd > 1e-12 else 1.0

        # loss_scale boosts Poisson when kx_nd is large
        self.loss_scale = float(np.clip(max(1.0, self.kx_nd/5.0), 1.0, 20.0))

        # krec: auto-balance D~R at Jsc conditions
        V_bi_nd = p.V_bi/p.E_g
        Gx_eff  = self.Gx_nd*self.Gx_scale
        X_avg   = Gx_eff/(self.kx_nd+self.kdiss_nd*V_bi_nd+1e-12)
        D_avg   = self.kdiss_nd*X_avg*V_bi_nd
        self.krec_nd = D_avg/0.25
        p.k_rec      = self.krec_nd/(self.n0*self.tau0)

        self.debye_ratio = p.q*self.n0*p.Height**2/(p.eps_si*p.E_g)
        self.V_bi_nd     = V_bi_nd
        self.J_scale     = self.n0*p.Dn*p.q/p.Height  # A/m²
        self.J_scale_phys = self.J_scale * 2.0 / (1.0 + self.mu_p_nd)
        self.n_min = max(1e-4, math.exp(-V_bi_nd/(2*self.VT_nd)))

        self._sanity()

    def _sanity(self):
        print("\n  Dimensionless params:")
        for name, val in [("debye",     self.debye_ratio),
                           ("Gx_nd",    self.Gx_nd),
                           ("kx_nd",    self.kx_nd),
                           ("kdiss_nd", self.kdiss_nd),
                           ("krec_nd",  self.krec_nd)]:
            ok = 1e-6 <= abs(val) <= 1e6
            print(f"    {name:10s} = {val:12.6f}  [{'OK' if ok else 'WARN'}]")
        print(f"    n_min      = {self.n_min:.2e}")
        print(f"    Gx_scale   = {self.Gx_scale:.6f}  (clamped <=1)")
        print(f"    loss_scale = {self.loss_scale:.2f}  "
              f"(Poisson phase1 weight = {20*self.loss_scale:.1f})")
        print(f"    J_scale    = {self.J_scale*0.1:.4f} mA/cm2/J_nd  (raw)")
        print(f"    J_scale_p  = {self.J_scale_phys*0.1:.4f} mA/cm2/J_nd  "
              f"(÷ max(1,{self.mu_p_nd:.1f}))")
        nm = self.n_min; V = self.V_bi_nd
        Jn = 1.0*(1.0*(-V)+(1.0-nm))
        Jp = self.mu_p_nd*(nm*(-V)+(1.0-nm))
        Jsc_bc = abs(Jn+Jp)*self.J_scale_phys*0.1
        print(f"    BC Jsc~    = {Jsc_bc:.4f} mA/cm2  (before training)\n")
        if self.kx_nd > 10.0:
            print(f"    *** tau_x WARNING: kx_nd={self.kx_nd:.1f} >> 1")
            print(f"    *** X≈0, photocurrent suppressed in PINN model")
            print(f"    *** GT may use direct generation — results not comparable")


# =============================================================================
# SECTION 3 — DATA LOADING
# =============================================================================
def load_data(data_dir):
    path=os.path.join(data_dir,'chem_morph_data.npy')
    print(f"Loading {path}...")
    raw=np.load(path,allow_pickle=True)
    N=len(raw)
    morphs=np.zeros((N,128,128),dtype=np.float32)
    params=np.zeros((N,5),dtype=np.float32)
    jsc=np.zeros(N,dtype=np.float32)
    for i,row in enumerate(raw):
        morphs[i]=np.array(row[0],dtype=np.float32)
        params[i]=np.array(row[1],dtype=np.float32)
        jsc[i]=float(row[2])
        if i%15000==0: print(f"  {i}/{N}")
    print(f"Loaded {N} | Jsc [{jsc.min():.3f},{jsc.max():.3f}] mA/cm²\n")
    return morphs,params,jsc


# =============================================================================
# SECTION 4 — MORPHOLOGY HANDLER
# Sigmoid steepness reduced from 20 to 10 for smoother interface transition
# =============================================================================
class MorphologyHandler:
    def __init__(self, grid, norm):
        from scipy.interpolate import RegularGridInterpolator
        H,W=grid.shape
        self.interp=RegularGridInterpolator(
            (np.linspace(0,1,H),np.linspace(0,1,W)),
            grid,method='linear',bounds_error=False,fill_value=0.5)
        self.norm=norm; self.morph_grid=grid
        gx=np.abs(np.gradient(grid,axis=0))
        gy=np.abs(np.gradient(grid,axis=1))
        print(f"  Morph {H}x{W} donor={float((grid>0.5).mean()):.3f} "
              f"intf={(gx+gy).mean():.4f}")

    def _phase(self,xy):
        p=self.interp(xy.detach().cpu().numpy()).astype(np.float32)
        # Steepness 10 (was 20): smoother transition, more physically realistic
        return torch.sigmoid(10*(torch.tensor(p).unsqueeze(1).to(xy.device)-0.5))

    def get_Gx(self,xy):
        return self.norm.Gx_nd*self.norm.Gx_scale*self._phase(xy)

    def get_mu_n(self,xy):
        d=self._phase(xy)
        return self.norm.mu_n_nd*((1-d)+self.norm.muRatio*d)

    def get_mu_p(self,xy):
        d=self._phase(xy)
        return self.norm.mu_p_nd*(d+self.norm.muRatio*(1-d))


# =============================================================================
# SECTION 5 — NETWORK (CHG 2: shared encoder + four output heads)
# =============================================================================
class OPV_PINN_2D(nn.Module):
    def __init__(self, norm, V_app_nd=0.0, nh=5, nn_=96):
        super().__init__()
        self.norm=norm; self.V_app=V_app_nd; self.n_min=norm.n_min

        # Shared encoder — learns spatial structure once for all four fields
        layers=[nn.Linear(2,nn_),nn.Tanh()]
        for _ in range(nh-1): layers+=[nn.Linear(nn_,nn_),nn.Tanh()]
        self.encoder  = nn.Sequential(*layers)

        # Four specialised output heads
        self.head_phi = nn.Linear(nn_,1)
        self.head_n   = nn.Linear(nn_,1)
        self.head_p   = nn.Linear(nn_,1)
        self.head_X   = nn.Linear(nn_,1)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self,xy):
        x=xy[:,0:1]; nm=self.n_min
        z=self.encoder(xy)
        phi=(self.norm.V_bi_nd/2)*(1-x)+(-self.norm.V_bi_nd/2+self.V_app)*x \
            +x*(1-x)*self.head_phi(z)
        n=nm*(1-x)+1.0*x+x*(1-x)*F.softplus(self.head_n(z))
        p=1.0*(1-x)+nm*x+x*(1-x)*F.softplus(self.head_p(z))
        X=x*(1-x)*F.softplus(self.head_X(z)+1.0)
        return phi,n,p,X


# =============================================================================
# SECTION 6 — RESIDUALS
# =============================================================================
def g2(f,xy):
    g=torch.autograd.grad(f,xy,torch.ones_like(f),
                           create_graph=True,retain_graph=True)[0]
    return g[:,0:1],g[:,1:2]


def residuals(model,xy,norm,morph):
    xy=xy.requires_grad_(True)
    phi,n,p,X=model(xy)
    px,py=g2(phi,xy); nx,ny=g2(n,xy)
    qx,qy=g2(p,xy);   Xx,Xy=g2(X,xy)
    pxx,_=g2(px,xy); _,pyy=g2(py,xy)
    Xxx,_=g2(Xx,xy); _,Xyy=g2(Xy,xy)
    Lphi=pxx+pyy; LX=Xxx+Xyy
    Em=torch.sqrt(px**2+py**2+1e-8)
    Gl=morph.get_Gx(xy); mn=morph.get_mu_n(xy); mp=morph.get_mu_p(xy)
    D=norm.kdiss_nd*X*Em; R=norm.krec_nd*n*p
    Jnx=mn*(n*px+nx); Jny=mn*(n*py+ny)
    Jpx=mp*(p*px-qx); Jpy=mp*(p*py-qy)
    dJnx,_=g2(Jnx,xy); _,dJny=g2(Jny,xy)
    dJpx,_=g2(Jpx,xy); _,dJpy=g2(Jpy,xy)
    rP=Lphi-norm.debye_ratio*(n-p)
    rn=(dJnx+dJny)-R+D
    rp=-(dJpx+dJpy)-R+D
    rX=norm.mu_x_nd*LX-norm.kx_nd*X-D+Gl
    return rP,rn,rp,rX


# =============================================================================
# SECTION 7 — LOSS
# CHG 3: get_w accepts explicit phase for adaptive curriculum
# CHG 4: interface-weighted residuals (4x at donor-acceptor boundaries)
# =============================================================================
def get_w(ep, n_epochs, loss_scale=1.0, phase=None):
    # CHG 3: explicit phase overrides epoch-based calculation
    if phase is None:
        f=ep/n_epochs
        phase=0 if f<0.20 else (1 if f<0.60 else 2)
    s=loss_scale
    if phase==0:
        return {'P':20.0*s,'n':0.01,'p':0.01,'X':0.1*s,'Jc':0.0}
    if phase==1:
        return {'P': 5.0*s,'n': 0.1,'p': 0.1,'X': 0.5, 'Jc':0.0}
    return     {'P': 1.0*s,'n': 1.0,'p': 1.0,'X': 1.0, 'Jc':10.0}


def loss_fn(model,xy,norm,morph,w):
    rP,rn,rp,rX=residuals(model,xy,norm,morph)

    # CHG 4: interface-weighted loss
    # 4*phase*(1-phase) peaks at 1.0 when phase=0.5 (at D-A boundary)
    # intf_weight = 1 in bulk, 4 at interface
    phase_vals  = morph._phase(xy).detach()
    intf_mask   = 4.0*phase_vals*(1.0-phase_vals)
    intf_weight = 1.0+3.0*intf_mask

    lP=w['P']*(rP**2*intf_weight).mean()
    ln=w['n']*(rn**2*intf_weight).mean()
    lp=w['p']*(rp**2*intf_weight).mean()
    lX=w['X']*(rX**2*intf_weight).mean()

    if w['Jc'] > 0:
        xy2=xy.detach().requires_grad_(True)
        phi2,n2,p2,_=model(xy2)
        px2,_=g2(phi2,xy2); nx2,_=g2(n2,xy2); qx2,_=g2(p2,xy2)
        Jnx2=norm.mu_n_nd*(n2*px2+nx2)
        Jpx2=norm.mu_p_nd*(p2*px2-qx2)
        Jt=Jnx2+Jpx2
        lJc=w['Jc']*(Jt.var()+0.5*((Jt-Jt.mean().detach())**2).mean())
    else:
        lJc=torch.tensor(0.0,device=xy.device)

    with torch.no_grad():
        _,n_v,p_v,_ = model(xy.detach())
    n_excess = F.relu(n_v - 1.2)
    p_excess = F.relu(p_v - 1.2)
    l_pile   = w['n']*(n_excess**2).mean() + w['p']*(p_excess**2).mean()

    total=lP+ln+lp+lX+lJc+l_pile
    return total,{'total':total.item(),'P':lP.item(),
                  'n':ln.item(),'p':lp.item(),'X':lX.item(),
                  'Jc':lJc.item(),'pile':l_pile.item()}


# =============================================================================
# SECTION 8 — COLLOCATION POINTS
# CHG 5: squared gradient magnitude + n_intf=600
# =============================================================================
def colloc(morph_handler,n_bulk=1000,n_intf=600):
    xy_b=torch.rand(n_bulk,2)
    mg=morph_handler.morph_grid; H,W=mg.shape

    # CHG 5: square gradient magnitude for aggressive interface focus
    grad_mag = np.abs(np.gradient(mg,axis=0))+np.abs(np.gradient(mg,axis=1))
    prob = (grad_mag**2).flatten()

    if prob.sum()>0:
        prob/=prob.sum()
        idx=np.random.choice(H*W,size=n_intf,p=prob,replace=True)
        xi=(idx//W/H+np.random.randn(n_intf)*0.01).clip(0.01,0.99)
        yi=(idx%W/W +np.random.randn(n_intf)*0.01).clip(0.01,0.99)
        xy_i=torch.tensor(np.stack([xi,yi],axis=1),dtype=torch.float32)
    else:
        xy_i=torch.rand(n_intf,2)

    n_bl=200
    xa=np.random.uniform(0.001,0.015,n_bl)
    xc=np.random.uniform(0.985,0.999,n_bl)
    ybl=np.random.rand(n_bl*2)
    xy_bl=torch.tensor(np.stack([np.concatenate([xa,xc]),ybl],axis=1),
                       dtype=torch.float32)
    return torch.cat([xy_b,xy_i,xy_bl],dim=0).to(DEVICE)


# =============================================================================
# SECTION 9 — TRAINING (CHG 3: adaptive curriculum)
# =============================================================================
def train(model,norm,morph,name="",n_epochs=20000,lr=5e-4,pe=2000):
    xy=colloc(morph)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    def lr_lambda(ep):
        warmup=500
        if ep<warmup: return ep/warmup
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/(n_epochs-warmup)))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)
    hist={k:[] for k in ['total','P','n','p','X','Jc','pile']}

    # CHG 3: adaptive phase — advances when loss actually converges
    phase=0

    def current_phase(ep, hist, phase):
        if phase==0 and ep>500:
            if len(hist['P'])>100 and np.mean(hist['P'][-100:])<0.1:
                return 1
        if phase==1 and ep>2000:
            if len(hist['n'])>100:
                if np.mean(hist['n'][-100:])+np.mean(hist['p'][-100:])<0.05:
                    return 2
        return phase

    print(f"\n{'='*60}")
    print(f"Training {name} | {n_epochs} epochs | {len(xy)} pts | lr={lr}")
    print(f"loss_scale={norm.loss_scale:.2f}  "
          f"Poisson_w_phase1={20*norm.loss_scale:.1f}")
    print(f"{'='*60}")
    print(f"{'Ep':>6}  {'Total':>9}  {'Poisson':>9}  "
          f"{'n':>8}  {'p':>8}  {'X':>8}  {'Jc':>8}  {'Ph':>4}  {'lr':>9}")
    print("-"*78)

    for ep in range(n_epochs):
        phase=current_phase(ep,hist,phase)
        opt.zero_grad()
        loss,losses=loss_fn(model,xy,norm,morph,
                            get_w(ep,n_epochs,norm.loss_scale,phase))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        opt.step(); sch.step()
        for k,v in losses.items(): hist[k].append(v)
        if ep%pe==0 or ep==n_epochs-1:
            print(f"{ep:>6}  {losses['total']:>9.3e}  {losses['P']:>9.3e}  "
                  f"{losses['n']:>8.3e}  {losses['p']:>8.3e}  "
                  f"{losses['X']:>8.3e}  {losses['Jc']:>8.3e}  "
                  f"{phase:>4}  {opt.param_groups[0]['lr']:.2e}")
    return hist


# =============================================================================
# SECTION 10 — JSC
# =============================================================================
def compute_jsc(model,norm,n_pts=300):
    model.eval()
    Jsc_vals=[]
    for x_val in [0.005,0.995]:
        xy=torch.cat([torch.full((n_pts,1),x_val),
                      torch.linspace(0.01,0.99,n_pts).unsqueeze(1)],
                     dim=1).to(DEVICE).requires_grad_(True)
        phi,n,p,X=model(xy)
        gp=torch.autograd.grad(phi,xy,torch.ones_like(phi),
                                create_graph=False,retain_graph=True)[0]
        gn=torch.autograd.grad(n,  xy,torch.ones_like(n),
                                create_graph=False,retain_graph=True)[0]
        gpp=torch.autograd.grad(p, xy,torch.ones_like(p),
                                 create_graph=False)[0]
        Jn=norm.mu_n_nd*(n*gp[:,0:1]+gn[:,0:1])
        Jp=norm.mu_p_nd*(p*gp[:,0:1]-gpp[:,0:1])
        Jt=(Jn+Jp).detach().cpu().numpy().flatten()
        Jsc_vals.append(abs(float(np.mean(Jt))))

    Jsc=float(np.mean(Jsc_vals))*norm.J_scale_phys*0.1

    xy2=torch.cat([torch.full((n_pts,1),0.995),
                   torch.linspace(0.01,0.99,n_pts).unsqueeze(1)],
                  dim=1).to(DEVICE).requires_grad_(True)
    phi2,n2,p2,_=model(xy2)
    gp2=torch.autograd.grad(phi2,xy2,torch.ones_like(phi2),
                             create_graph=False,retain_graph=True)[0]
    gn2=torch.autograd.grad(n2,  xy2,torch.ones_like(n2),
                             create_graph=False,retain_graph=True)[0]
    gpp2=torch.autograd.grad(p2, xy2,torch.ones_like(p2),
                              create_graph=False)[0]
    Jn2=norm.mu_n_nd*(n2*gp2[:,0:1]+gn2[:,0:1])
    Jp2=norm.mu_p_nd*(p2*gp2[:,0:1]-gpp2[:,0:1])
    Jt2=(Jn2+Jp2).detach().cpu().numpy().flatten()
    latvar=float(np.std(Jt2)/(abs(np.mean(Jt2))+1e-12))

    print(f"  Jsc at anode  : {Jsc_vals[0]*norm.J_scale_phys*0.1:.4f} mA/cm2")
    print(f"  Jsc at cathode: {Jsc_vals[1]*norm.J_scale_phys*0.1:.4f} mA/cm2")
    print(f"  Jsc (average) : {Jsc:.4f} mA/cm2")
    return Jsc,Jt2,latvar


# =============================================================================
# SECTION 11 — GRID EVAL
# =============================================================================
def eval_grid(model,norm,gs=64):
    model.eval()
    XX,YY=np.meshgrid(np.linspace(0.01,0.99,gs),
                      np.linspace(0.01,0.99,gs),indexing='ij')
    flat=torch.tensor(np.stack([XX.flatten(),YY.flatten()],axis=1),
                      dtype=torch.float32).to(DEVICE)
    pl,nl,ppl,Xl,Jl=[],[],[],[],[]
    for i in range(0,len(flat),512):
        b=flat[i:i+512].requires_grad_(True)
        ph,nb,pb,Xb=model(b)
        gp=torch.autograd.grad(ph,b,torch.ones_like(ph),
                                create_graph=False,retain_graph=True)[0]
        gn=torch.autograd.grad(nb,b,torch.ones_like(nb),
                                create_graph=False,retain_graph=True)[0]
        gpp=torch.autograd.grad(pb,b,torch.ones_like(pb),
                                 create_graph=False)[0]
        Jnx=norm.mu_n_nd*(nb*gp[:,0:1]+gn[:,0:1])
        Jpx=norm.mu_p_nd*(pb*gp[:,0:1]-gpp[:,0:1])
        pl.append(ph.detach().cpu().numpy()); nl.append(nb.detach().cpu().numpy())
        ppl.append(pb.detach().cpu().numpy()); Xl.append(Xb.detach().cpu().numpy())
        Jl.append((Jnx+Jpx).detach().cpu().numpy())
    r=lambda l: np.concatenate(l).reshape(gs,gs)
    return r(pl),r(nl),r(ppl),r(Xl),r(Jl)


# =============================================================================
# SECTION 12 — PLOT
# =============================================================================
def plot_fields(morph,phi,n,p,X,Jx,hist,Jsc,gt,lv,name,norm,save_dir="."):
    fig=plt.figure(figsize=(16,8))
    gs_=gridspec.GridSpec(2,4,hspace=0.4,wspace=0.35)
    err=f" err={abs(Jsc-gt)/gt*100:.1f}%" if gt>0 else ""
    fig.suptitle(f'{name}  PINN={Jsc:.4f}  GT={gt:.4f} mA/cm²{err}  '
                 f'(latvar={lv:.3f})',fontsize=11,fontweight='bold')
    mg=morph.morph_grid; ext=[0,100,0,100]
    def ct(ax):
        ax.contour(np.linspace(0,100,mg.shape[0]),
                   np.linspace(0,100,mg.shape[1]),
                   mg.T,levels=[0.5],colors='white',linewidths=0.7,alpha=0.6)
    def hm(ax,d,t,c,vn=None,vx=None):
        im=ax.imshow(d.T,origin='lower',cmap=c,extent=ext,aspect='auto',
                     vmin=vn,vmax=vx)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04); ct(ax)
        ax.set_title(t,fontsize=9); ax.set_xlabel('x(nm)',fontsize=8)
        ax.set_ylabel('y(nm)',fontsize=8); ax.tick_params(labelsize=7)
    ax0=fig.add_subplot(gs_[0,0])
    im=ax0.imshow(mg.T,origin='lower',cmap='RdYlBu',extent=ext,
                  aspect='auto',vmin=0,vmax=1)
    plt.colorbar(im,ax=ax0,fraction=0.046,pad=0.04)
    ax0.set_title('Morphology',fontsize=9)
    ax0.set_xlabel('x(nm)',fontsize=8); ax0.set_ylabel('y(nm)',fontsize=8)
    hm(fig.add_subplot(gs_[0,1]),phi*norm.V0,'φ (V)','RdBu_r')
    hm(fig.add_subplot(gs_[0,2]),X,'Exciton X','YlOrBr')
    hm(fig.add_subplot(gs_[1,0]),n,'Electron n','Greens')
    hm(fig.add_subplot(gs_[1,1]),p,'Hole p','Oranges')
    vm=max(abs(Jx.min()),abs(Jx.max()))
    hm(fig.add_subplot(gs_[1,2]),Jx,'Jx (interior)','RdBu_r',-vm,vm)
    ax6=fig.add_subplot(gs_[:,3])
    for k,c,lb in [('total','k','total'),('P','#3B8BD4','Poisson'),
                    ('n','#1D9E75','n'),('p','#E85D24','p'),
                    ('X','#BA7517','X'),('Jc','#534AB7','Jcons')]:
        if k in hist and any(v>0 for v in hist[k]):
            ax6.semilogy(hist[k],color=c,lw=1.5 if k=='total' else 1,label=lb)
    ax6.axhline(0.01,color='gray',lw=1,ls='--',alpha=0.5,label='target')
    ax6.legend(fontsize=7)
    ax6.set_xlabel('Epoch',fontsize=8); ax6.set_ylabel('Loss',fontsize=8)
    ax6.set_title('Loss (target<0.01)',fontsize=9)
    ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)
    fname=os.path.join(save_dir,f'opv_v6_{name}.png')
    plt.savefig(fname,dpi=150,bbox_inches='tight'); plt.show()
    print(f"Saved: {fname}")


def plot_scatter(results,save_dir="."):
    gt=[r['gt'] for r in results]; pn=[r['pinn'] for r in results]
    ers=[r['err'] for r in results]; idx=[r['idx'] for r in results]
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    fig.suptitle(f'PINN v6 vs Dataset — {len(results)} morphologies  '
                 f'mean={np.mean(ers):.1f}%',fontsize=12,fontweight='bold')
    ax=axes[0]
    sc=ax.scatter(gt,pn,s=60,c=ers,cmap='RdYlGn_r',vmin=0,vmax=50,
                  edgecolors='gray',lw=0.5,zorder=3)
    plt.colorbar(sc,ax=ax,label='Error %')
    for r in results:
        ax.annotate(str(r['idx']),(r['gt'],r['pinn']),fontsize=7,
                    xytext=(3,3),textcoords='offset points')
    lims=[min(gt+pn)*0.85,max(gt+pn)*1.15]
    ax.plot(lims,lims,'k--',lw=1.2,alpha=0.6,label='perfect')
    ax.fill_between(lims,[v*0.75 for v in lims],[v*1.25 for v in lims],
                    alpha=0.08,color='gray',label='±25%')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xlabel('GT Jsc (mA/cm²)',fontsize=11)
    ax.set_ylabel('PINN Jsc (mA/cm²)',fontsize=11)
    ax.set_title('Jsc comparison',fontsize=11); ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax2=axes[1]
    cols=['#1D9E75' if e<25 else '#E85D24' for e in ers]
    ax2.bar(range(len(ers)),ers,color=cols,alpha=0.8,edgecolor='white')
    ax2.axhline(25,color='#E85D24',lw=1,ls='--',label='25%')
    ax2.axhline(np.mean(ers),color='black',lw=1.5,
                label=f'mean={np.mean(ers):.1f}%')
    ax2.set_xticks(range(len(idx)))
    ax2.set_xticklabels([f'#{i}' for i in idx],fontsize=8)
    ax2.set_xlabel('Morphology',fontsize=11)
    ax2.set_ylabel('Error (%)',fontsize=11)
    ax2.set_title('Per-morphology error',fontsize=11); ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    fname=os.path.join(save_dir,'scatter_v6.png')
    plt.savefig(fname,dpi=150,bbox_inches='tight'); plt.show()
    print(f"Saved: {fname}")


# =============================================================================
# SECTION 13 — VALIDATION
# =============================================================================
def validate(morphs,params,jsc_gt,indices,n_epochs,save_dir):
    results=[]
    for idx in indices:
        gt=float(jsc_gt[idx])
        print(f"\n{'='*55}\nMorphology #{idx}  GT={gt:.4f}\n{'='*55}")
        phys=PhysicalParams(); phys.update_from_dataset(params[idx])
        norm=NormParams(phys)
        morph=MorphologyHandler(morphs[idx],norm)
        model=OPV_PINN_2D(norm).to(DEVICE)
        hist=train(model,norm,morph,name=f"m{idx}",n_epochs=n_epochs,lr=5e-4)
        Jsc,Jt,lv=compute_jsc(model,norm)
        err=abs(Jsc-gt)/(gt+1e-8)*100
        results.append(dict(idx=idx,gt=gt,pinn=Jsc,err=err,lv=lv,
                            loss=hist['total'][-1]))
        print(f"  PINN={Jsc:.4f}  GT={gt:.4f}  err={err:.1f}%  "
              f"lv={lv:.4f}  loss={hist['total'][-1]:.3e}")
        phi2,n2,p2,X2,Jx2=eval_grid(model,norm)
        plot_fields(morph,phi2,n2,p2,X2,Jx2,hist,Jsc,gt,lv,
                    f"m{idx}",norm,save_dir)
    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    print(f"{'#':>5}  {'GT':>8}  {'PINN':>8}  {'Err%':>7}  {'Loss':>10}")
    for r in results:
        print(f"{r['idx']:>5}  {r['gt']:>8.4f}  {r['pinn']:>8.4f}  "
              f"{r['err']:>6.1f}%  {r['loss']:>10.3e}  "
              f"{'✓' if r['err']<25 else '✗'}")
    ers=[r['err'] for r in results]
    print(f"\nMean={np.mean(ers):.1f}%  Within25%="
          f"{sum(1 for e in ers if e<25)}/{len(ers)}")
    plot_scatter(results,save_dir)
    return results


# =============================================================================
# SECTION 14 — MAIN
# =============================================================================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='.')
    ap.add_argument('--morph_idx',type=int,default=0)
    ap.add_argument('--n_epochs', type=int,default=20000)
    ap.add_argument('--validate', type=int,default=0)
    ap.add_argument('--save_dir', default='.')
    args=ap.parse_args(); os.makedirs(args.save_dir,exist_ok=True)

    print("="*55)
    print("OPV 2D PINN v6 — shared encoder + interface weighting")
    print("="*55)

    morphs,params,jsc=load_data(args.data_dir)

    if args.validate>0:
        idx=list(np.linspace(0,len(jsc)-1,args.validate,dtype=int))
        print(f"Validating: {idx}")
        validate(morphs,params,jsc,idx,args.n_epochs,args.save_dir)
    else:
        i=args.morph_idx; gt=float(jsc[i])
        phys=PhysicalParams(); phys.update_from_dataset(params[i])
        norm=NormParams(phys)
        print(f"\nMorphology #{i}  GT={gt:.4f} mA/cm²")
        morph=MorphologyHandler(morphs[i],norm)
        model=OPV_PINN_2D(norm).to(DEVICE)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
        hist=train(model,norm,morph,name=f"m{i}",
                   n_epochs=args.n_epochs,lr=5e-4)
        Jsc,Jt,lv=compute_jsc(model,norm)
        err=abs(Jsc-gt)/(gt+1e-8)*100
        print(f"\n{'='*55}")
        print(f"PINN Jsc  = {Jsc:.4f} mA/cm²")
        print(f"GT Jsc    = {gt:.4f} mA/cm²")
        print(f"Error     = {err:.1f}%")
        print(f"Lat. var. = {lv:.4f}  (good if <0.15)")
        print(f"Loss      = {hist['total'][-1]:.4e}  (good if <0.01)")
        print(f"{'='*55}")
        phi2,n2,p2,X2,Jx2=eval_grid(model,norm)
        print(f"\nField ranges:")
        print(f"  phi [{phi2.min()*norm.V0:.3f},{phi2.max()*norm.V0:.3f}] V")
        print(f"  n   [{n2.min():.4f},{n2.max():.4f}]")
        print(f"  p   [{p2.min():.4f},{p2.max():.4f}]")
        print(f"  X   [{X2.min():.4f},{X2.max():.4f}]")
        plot_fields(morph,phi2,n2,p2,X2,Jx2,hist,Jsc,gt,lv,
                    f"m{i}",norm,args.save_dir)
    print("\nDone.")

if __name__=="__main__":
    main()