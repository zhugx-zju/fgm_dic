"""Preserved verified material/test-space probe core, originally developed in the isolated 20261021 study.

Numerical functions below are unchanged. It reuses the production sparse
assembly, observations, material tensors and regularizers. Units: mm, N, MPa.
This probe retains the original fixed ROI/nu; it is not a general SUPG method.
"""
from __future__ import annotations
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
from fgm_asm.mesh import MeshInfo,setup_boundary_conditions
from fgm_asm.material import MaterialInfo
from fgm_asm.fem_forward import assemble_global_stiffness,fem_assemble,forward_solver
from fgm_asm.observation import build_observation_package
from fgm_asm.regularization import get_huber_tv_value_gradient,get_tikhonov_regularization,get_tikhonov_gradient
from fgm_asm.l_curve import select_l_curve_corner
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/dic_full/load380_20261022/spaces'
L,H,THICKNESS,NU=105.99508453656999,18.48415643522,3.2,.26
BOUNDARIES=np.array([15.74727501954+16*i for i in range(6)])
METHODS=('Q1_H1','Q1_TV','P0_TV','Q1_H1_PG','P0_TV_PG')


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=plain), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def quadrature(mesh, alpha):
    """Conforming test enrichment psi=N+alpha*hx*b*dN/dx; b=(1-s²)(1-t²).

    b=0 on every edge, so boundary traction values equal the Galerkin values.
    Body force is zero. This is bubble PG, NOT a derived 2D SUPG method.
    Three-point Gauss quadrature exactly integrates this polynomial bilinear form.
    """
    a, w = np.polynomial.legendre.leggauss(3)
    s, t = np.meshgrid(a, a)
    s, t = s.ravel(), t.ravel()
    weights = np.outer(w, w).ravel()*mesh.el*mesh.eh/4
    sx, sy = np.array([-1,1,1,-1]), np.array([-1,-1,1,1])
    N = (1+s[:,None]*sx)*(1+t[:,None]*sy)/4
    dx = sx*(1+t[:,None]*sy)/(2*mesh.el)
    dy = sy*(1+s[:,None]*sx)/(2*mesh.eh)
    b = (1-s*s)*(1-t*t)
    bx = -4*s*(1-t*t)/mesh.el
    by = -4*t*(1-s*s)/mesh.eh
    wx = dx+alpha*mesh.el*bx[:,None]*dx
    wy = dy+alpha*mesh.el*(by[:,None]*dx+b[:,None]*(sx*sy/(mesh.el*mesh.eh)))
    def strain_matrix(gx, gy):
        B = np.zeros((9,3,8))
        B[:,0,0::2], B[:,1,1::2] = gx, gy
        B[:,2,0::2], B[:,2,1::2] = gy, gx
        return B
    Bu, Bw = strain_matrix(dx,dy), strain_matrix(wx,wy)
    D = MaterialInfo(NU).get_elasticity_matrix()
    unit = np.einsum('gia,ij,gjb,g->gab', Bw,D,Bu,weights)
    x = (mesh.x_ele@N.T).T
    y = (mesh.y_ele@N.T).T
    return N, weights, unit, x, y

class Data:
    def __init__(self, mesh, xy, d, areas, force):
        self.obs = build_observation_package(mesh, xy[:,0],xy[:,1],d[:,0],d[:,1],areas, sampling_mode='physical_points')
        self.bc = setup_boundary_conditions(mesh,L,H,force*H/20,THICKNESS)
        self.H = self.obs.H
        self.d = self.obs.u_measured
        self.w = self.obs.weight_vector
        x,y = xy[:,0]-xy[:,0].mean(),xy[:,1]-xy[:,1].mean()
        R = np.zeros((2*len(x),3)); R[0::2,0]=1; R[1::2,1]=1
        R[0::2,2]=-y; R[1::2,2]=x
        self.R=R
        self.Qinv=np.linalg.inv(R.T@(self.w[:,None]*R))
        self.RW=R.T*self.w
    def evaluate(self,u):
        r0=self.d-self.H@u
        c=self.Qinv@(self.RW@r0)
        r=r0-self.R@c
        return .5*float(r@(self.w*r)), np.asarray(self.H.T@(self.w*r)).ravel(), r, c

class Model:
    def __init__(self,mesh,method,bc):
        self.mesh,self.method,self.bc=mesh,method,bc
        self.p0=method.startswith('P0')
        self.alpha=.25 if method.endswith('_PG') else 0.
        self.N,self.weights,self.unit,self.xg,self.yg=quadrature(mesh,self.alpha)
        self.n=mesh.n_el if self.p0 else mesh.n_nod
        self.free=np.setdiff1d(np.arange(mesh.n_dof),bc['fixdof'])
        ix=np.arange(mesh.n_el).reshape(mesh.nel_y,mesh.nel_x)
        self.a=np.r_[ix[:,:-1].ravel(),ix[:-1,:].ravel()]
        self.b=np.r_[ix[:,1:].ravel(),ix[1:,:].ravel()]
        self.face=np.r_[np.full(mesh.nel_y*(mesh.nel_x-1),mesh.eh),np.full((mesh.nel_y-1)*mesh.nel_x,mesh.el)]
        self.delta=np.r_[np.full(mesh.nel_y*(mesh.nel_x-1),10*mesh.el),np.full((mesh.nel_y-1)*mesh.nel_x,10*mesh.eh)]
        self.material=MaterialInfo(NU)
    def gauss_values(self,E):
        return np.broadcast_to(E,(9,len(E))) if self.p0 else (E[self.mesh.ele_nods_id-1]@self.N.T).T
    def stiffness(self,E=None,gauss=None):
        Eg=self.gauss_values(E) if gauss is None else gauss
        ke=np.einsum('ge,gab->eab',Eg,self.unit)
        return assemble_global_stiffness(self.mesh,ke)
    def forward(self,E=None,gauss=None):
        A=self.stiffness(E,gauss)
        lu=splu(A[self.free][:,self.free].tocsc(),permc_spec='COLAMD' if self.alpha else 'MMD_AT_PLUS_A')
        u=np.zeros(self.mesh.n_dof)
        u[self.free]=lu.solve(self.bc['force'][self.free])
        return u,lu,A
    def regularization(self,E):
        if self.p0:
            diff=E[self.a]-E[self.b]; mag=np.abs(diff)
            rho=np.where(mag<=self.delta,diff**2/(2*self.delta),mag-self.delta/2)
            v=self.face*diff/np.maximum(mag,self.delta)
            g=np.bincount(self.a,weights=v,minlength=self.n)-np.bincount(self.b,weights=v,minlength=self.n)
            return float(self.face@rho),g
        self.material.update(E)
        if '_H1' in self.method:
            return get_tikhonov_regularization(self.mesh,self.material),get_tikhonov_gradient(self.mesh,self.material)
        return get_huber_tv_value_gradient(self.mesh,self.material,delta=10.)
    def value_gradient(self,m,data,gamma):
        E=np.exp(m)
        if not np.all(np.isfinite(E)&(E>0)): raise FloatingPointError('Nonfinite/nonpositive E')
        u,lu,_=self.forward(E)
        cost,rhs,r,_=data.evaluate(u)
        lam=np.zeros_like(u)
        # A is non-symmetric for PG. The transpose solve is mandatory.
        lam[self.free]=lu.solve(rhs[self.free],trans='T')
        elemental=np.einsum('ea,gab,eb->ge',lam[self.mesh.ele_dof_id],self.unit,u[self.mesh.ele_dof_id])
        if self.p0: gd=elemental.sum(axis=0)
        else:
            local=elemental.T@self.N
            gd=np.bincount((self.mesh.ele_nods_id-1).ravel(),weights=local.ravel(),minlength=self.n)
        reg,gr=self.regularization(E)
        grad=(gd+.5*gamma*gr)*E
        return dict(value=cost+.5*gamma*reg,gradient=grad,E=E,u=u,data_cost=cost,reg=reg,r=r)

def gradient_checks():
    mesh=MeshInfo(L,H,8,3)
    xy=mesh.coord
    areas=np.ones(len(xy))*L*H/len(xy)
    bc=setup_boundary_conditions(mesh,L,H,119.77617870327234*H/20,THICKNESS)
    base=Model(mesh,'Q1_H1',bc)
    E=1800+150*np.cos(mesh.X/L*2*np.pi)+50*mesh.Y/H
    u,_,A=base.forward(E)
    mat=MaterialInfo(NU);mat.update(E)
    fem=fem_assemble(mesh,mat,bc);ref=forward_solver(fem)
    rows=[]
    rng=np.random.default_rng(20261021)
    data=Data(mesh,xy,np.column_stack([u[0::2],u[1::2]])+rng.normal(0,1e-4,(len(xy),2)),areas,119.77617870327234)
    for method in METHODS:
        model=Model(mesh,method,bc)
        m=np.log(1800+100*rng.normal(size=model.n))
        direction=rng.normal(size=model.n);direction/=np.linalg.norm(direction)
        gamma=1e-9 if '_H1' in method else 1e-7
        st=model.value_gradient(m,data,gamma)
        analytic=float(st['gradient']@direction)
        errors=[]
        for h in (1e-3,1e-4,1e-5):
            fd=(model.value_gradient(m+h*direction,data,gamma)['value']-model.value_gradient(m-h*direction,data,gamma)['value'])/(2*h)
            errors.append(abs(fd-analytic)/max(abs(fd),abs(analytic),1e-20))
        uniform,_,Ap=model.forward(np.full(model.n,1800.))
        # sigma=F/(20*t); three rigid gauges give u_x=sigma*x/E, u_y=-nu*sigma*y/E.
        sig=119.77617870327234/(20*THICKNESS)
        exact=np.column_stack([sig/1800*mesh.X,-NU*sig/1800*mesh.Y]).ravel()
        row=dict(method=method,relative_directional_errors=errors,best_error=min(errors),uniform_max_error_mm=float(abs(uniform-exact).max()),asymmetry=float(scipy.sparse.linalg.norm(Ap-Ap.T)/scipy.sparse.linalg.norm(Ap)))
        assert min(errors)<2e-5,row
        assert row['uniform_max_error_mm']<1e-9,row
        rows.append(row)
    result=dict(checks=rows,baseline_stiffness_relative_error=float(scipy.sparse.linalg.norm(A-fem.K)/scipy.sparse.linalg.norm(fem.K)),baseline_displacement_error_mm=float(abs(ref-u).max()),quadrature='3x3 for all methods; baseline checked against existing 2x2 kernel',test_loads='bubble vanishes on all edges; zero body force, so traction RHS is unchanged',passed=True)
    assert result['baseline_stiffness_relative_error']<1e-12
    save_json(OUT/'verification.json',result)
    print(json.dumps(result,default=plain),flush=True)

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def fit_uniform(model,data):
    unit,_,_=model.forward(np.ones(model.n))
    measured=data.evaluate(np.zeros_like(unit))[2]
    pred=measured-data.evaluate(unit)[2]
    return float(pred@(data.w*pred)/(pred@(data.w*measured)))

def scan(model,data,folder,count,gamma_override=None):
    folder.mkdir(exist_ok=False)
    initial=fit_uniform(model,data)
    gammas=(np.logspace(-12,-4,count) if '_H1' in model.method else np.logspace(-10,-2,count)) if gamma_override is None else np.asarray(gamma_override)
    m=np.full(model.n,np.log(initial));rows=[]
    cache={}
    for i,gamma in enumerate(gammas):
        start=time.perf_counter()
        def objective(x):
            st=model.value_gradient(x,data,gamma)
            cache['x'],cache['state']=x.copy(),st
            return st['value'],st['gradient']
        opt=minimize(objective,m,jac=True,method='L-BFGS-B',options=dict(maxiter=1500,maxfun=12000,ftol=1e-30,gtol=1e-12,maxls=50,maxcor=10))
        st=cache['state'] if np.array_equal(cache['x'],opt.x) else model.value_gradient(opt.x,data,gamma)
        np.savez_compressed(folder/f'gamma_{i:02d}.npz',E=st['E'],u=st['u'],gradient=st['gradient'])
        row=dict(gamma=float(gamma),rho=float(np.sqrt(2*st['data_cost'])),eta=float(np.sqrt(max(st['reg'],0))),success=bool(opt.success),gtol=bool(abs(st['gradient']).max()<=1e-12),gradient_inf=float(abs(st['gradient']).max()),iterations=int(opt.nit),evaluations=int(opt.nfev),message=str(opt.message),elapsed_s=time.perf_counter()-start)
        rows.append(row);write_csv(folder/'scan.csv',rows)
        if opt.success:m=opt.x.copy()
        print(folder.parent.name,model.method,i+1,'/',count,'gamma',f'{gamma:.3g}','rho',f'{row["rho"]:.4g}','ok',opt.success,'nit',opt.nit,flush=True)
    selection=select_l_curve_corner(gammas,[r['rho'] for r in rows],[r['eta'] for r in rows],[r['success'] for r in rows])
    selection['initial_uniform_E_MPa']=initial
    save_json(folder/'selection.json',selection)
    return selection

def metrics(model,E,data,truth):
    Eg=model.gauss_values(E)
    region=np.searchsorted(BOUNDARIES,model.xg,side='right')
    fullw=np.broadcast_to(model.weights[:,None],Eg.shape)
    means=np.array([np.sum((fullw*Eg)[region==k])/np.sum(fullw[region==k]) for k in range(7)])
    yg=Eg.reshape(9,model.mesh.nel_y,model.mesh.nel_x)
    ystd=np.std(yg,axis=1)
    out=dict(regional_mape_percent=float(np.mean(abs(means-truth)/truth)*100),regional_means_MPa=means,relative_y_spatial_sd_percent=float(np.sqrt(np.mean(ystd**2))/np.mean(Eg)*100),min_E_MPa=float(E.min()),max_E_MPa=float(E.max()))
    if data is not None:
        u,_,_=model.forward(E);_,_,r,_=data.evaluate(u)
        out['ux_rmse_um']=float(np.sqrt(np.average(r[0::2]**2,weights=data.w[0::2]))*1000)
        out['uy_rmse_um']=float(np.sqrt(np.average(r[1::2]**2,weights=data.w[1::2]))*1000)
    target=truth[region]
    out['gauss_relative_rms_percent']=float(np.sqrt(np.sum(fullw*(Eg-target)**2)/np.sum(fullw*target**2))*100)
    return out
