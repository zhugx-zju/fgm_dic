"""Read-only figures and HTML for the fixed-observation mesh experiment."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from fgm_asm.mesh import MeshInfo
from fgm_asm.observation import build_observation_package
from lib.exact_tv_mesh_study import DEFAULT_OUTPUT, MESHES, dense_metrics, fixed_case, read_json
from lib.exact_tv_study import field_metrics, save_json, truth_function


NAMES={'h1':'梯度平方正则','exact_tv':'未平滑TV正则'}
COLORS={'h1':'#0072B2','exact_tv':'#D55E00'}
STYLES={12:':',24:'--',48:'-'}


def width_label(value):
    return f'{value:.3f}' if value is not None else '—'


def collect(root):
    result=[]
    for kind in ('step','smooth'):
        for nx,(_,ny) in MESHES.items():
            for method in ('h1','exact_tv'):
                directory=root/f'{nx}x{ny}/{kind}_{method}'
                if not (directory/'selected_metrics.json').exists():
                    continue
                row=read_json(directory/'selected_metrics.json')
                if (directory/'validation_metrics.json').exists():
                    row['validation']=read_json(directory/'validation_metrics.json')
                result.append(row)
    return result


def selected_field(root,row):
    directory=root/f"{row['nx']}x{row['ny']}/{row['kind']}_{row['method']}"
    selection=read_json(directory/'selection.json')
    with np.load(directory/'fields.npz') as saved:
        return saved['E'][selection['optimal_index']].copy()


def sample(mesh,field,x,y):
    zeros=np.zeros_like(x)
    matrix=build_observation_package(mesh,x,y,zeros,zeros,np.ones_like(x),
                                    sampling_mode='physical_points').H[::2,::2]
    return np.asarray(matrix@field).ravel()


def save_figure(fig,root,name):
    for ext in ('png','svg'):
        fig.savefig(root/f'{name}.{ext}',dpi=220)
    plt.close(fig)


def representation_bounds(root):
    path=root/'representation_bounds.json'
    if path.exists():return read_json(path)
    grid=MeshInfo(30.,8.,96,32);nodes=grid.ele_nods_id-1
    x=np.einsum('ga,ea->ge',grid.gauss_N,grid.X[nodes]).ravel()
    y=np.einsum('ga,ea->ge',grid.gauss_N,grid.Y[nodes]).ravel()
    w=(grid.gauss_abs_det_j*grid.gauss_w[:,None]).ravel()
    rows=[]
    for nx in MESHES:
        mesh=MeshInfo(30.,8.,*MESHES[nx])
        zeros=np.zeros_like(x)
        matrix=build_observation_package(mesh,x,y,zeros,zeros,w,
            sampling_mode='physical_points').H[::2,::2]
        mass=(matrix.T@diags(w)@matrix).tocsc()
        for kind in ('smooth','step'):
            truth=truth_function(kind)(x,y)
            best=spsolve(mass,matrix.T@(w*truth))
            error=matrix@best-truth
            rows.append(dict(nx=nx,kind=kind,
                dense_best_L2_relative_rms=float(np.sqrt(w@error**2/(w@truth**2))),
                minimum_modulus_MPa=float(best.min()),uses_truth_only_for_representation_bound=True,
                not_an_inverse_result=True))
    save_json(path,rows)
    return rows


def fixed_gamma_diagnostic(root,lookup):
    rows=[]
    for kind in ('step','smooth'):
        for method in ('h1','exact_tv'):
            gamma=lookup[(kind,12,method)]['selected_gamma']
            if gamma is None:continue
            for nx in MESHES:
                directory=root/f'{nx}x{MESHES[nx][1]}/{kind}_{method}'
                scan=read_json(directory/'scan.json')
                matches=[i for i,r in enumerate(scan) if np.isclose(r['gamma'],gamma,rtol=1e-12,atol=0)]
                if len(matches)!=1:raise ValueError('Expected the same gamma grid at every mesh')
                i=matches[0];row=dict(nx=nx,kind=kind,method=method,gamma=gamma,
                                     converged=scan[i]['converged'])
                if row['converged']:
                    with np.load(directory/'fields.npz') as saved:field=saved['E'][i]
                    mesh,case,_=fixed_case(root,nx,kind)
                    row.update(field_metrics(mesh,case,field,kind))
                    row.update(dense_metrics(mesh,field,kind))
                rows.append(row)
    save_json(root/'fixed_gamma_metrics.json',rows)
    return rows


def render(root):
    rows=collect(root)
    if len(rows)!=12:
        raise ValueError(f'Need all 12 scan outcomes before rendering; found {len(rows)}')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,
                         'axes.unicode_minus':False,'svg.fonttype':'none'})
    lookup={(r['kind'],r['nx'],r['method']):r for r in rows}
    fixed=fixed_gamma_diagnostic(root,lookup)
    save_json(root/'summary.json',rows)
    flat=[]
    for row in rows:
        for val in row.get('validation',[]):
            flat.append({k:v for k,v in val.items() if not isinstance(v,(list,dict))})
    if flat:
        keys=list(dict.fromkeys(k for row in flat for k in row))
        with (root/'validation_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(flat)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
    fig.suptitle('固定833个观测点和同一份噪声，只改变反演网格\n'
                 '每档独立L-curve选γ；误差线为固定γ后两个新噪声样本的范围',fontsize=12)
    for ax,kind in zip(axes,('step','smooth')):
        for method in ('h1','exact_tv'):
            x,means,low,high=[],[],[],[]
            for pos,nx in enumerate(MESHES):
                row=lookup[(kind,nx,method)]
                valid=[v for v in row.get('validation',[]) if v['converged']]
                if row['status']!='selected' or len(valid)!=2:continue
                values=np.array([v['area_relative_rms']*100 for v in valid])
                x.append(pos);means.append(values.mean());low.append(values.mean()-values.min());high.append(values.max()-values.mean())
            ax.errorbar(x,means,yerr=[low,high],color=COLORS[method],
                linestyle='--' if method=='h1' else '-',marker='o',capsize=4,label=NAMES[method])
        ax.set(xticks=range(3),xticklabels=['12×4\n65个节点E','24×8\n225个节点E','48×16\n833个节点E'],
               title='斜界面阶梯场' if kind=='step' else '平滑二维场',ylabel='模量面积加权相对RMS（%）')
        ax.set_ylim(bottom=0);ax.legend(frameon=False)
    save_figure(fig,root,'mesh_error')

    formula=('两种方法均为独立节点Q4模量，位移试函数与测试函数均为 $N_i$\n'
        r'$J_{H1}=\frac{1}{2}r^TW_A r+\frac{\gamma}{2}\int(E_x^2+E_y^2)dA$'
        '     '
        r'$J_{TV}=\frac{1}{2}r^TW_A r+\gamma\int\sqrt{E_x^2+E_y^2}dA$')
    fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    fig.suptitle(formula,fontsize=12)
    xx,yy=np.meshgrid(np.linspace(0,30,121),np.linspace(0,8,33))
    fields={}
    for method in ('h1','exact_tv'):
        for nx in MESHES:
            row=lookup[('step',nx,method)]
            if row['status']=='selected':
                mesh,_,_=fixed_case(root,nx,'step')
                fields[(nx,method)]=sample(mesh,selected_field(root,row),xx.ravel(),yy.ravel()).reshape(xx.shape)
    vmin=100*np.floor(min(np.min(v) for v in fields.values())/100)
    vmax=100*np.ceil(max(np.max(v) for v in fields.values())/100)
    for row_id,method in enumerate(('h1','exact_tv')):
        for col,nx in enumerate(MESHES):
            ax=axes[row_id,col];row=lookup[('step',nx,method)]
            if row['status']!='selected':
                ax.text(.5,.5,'未完成选参：'+row['status'],ha='center',transform=ax.transAxes)
                ax.set_axis_off();continue
            image=ax.pcolormesh(xx,yy,fields[(nx,method)],shading='auto',cmap='viridis',vmin=vmin,vmax=vmax)
            y=np.linspace(0,8,100);ax.plot(13.5-.6*y,y,'w--',linewidth=1)
            ax.set_aspect('equal');ax.set(xlabel='x（mm）',ylabel='y（mm）',
                title=f"{NAMES[method]} / {nx}×{MESHES[nx][1]}\nγ={row['selected_gamma']:.2e}；误差={100*row['area_relative_rms']:.2f}%")
    fig.colorbar(image,ax=axes.ravel().tolist(),label='E（MPa）',shrink=.7)
    save_figure(fig,root,'step_fields')

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained',sharey=True)
    fig.suptitle('阶梯场中线 y=4 mm；γ由各网格独立L-curve选出',fontsize=12)
    x=np.linspace(0,30,1201)
    for ax,method in zip(axes,('h1','exact_tv')):
        ax.plot([0,11.1,11.1,30],[2400,2400,1400,1400],color='black',label='真值',linewidth=1.6)
        for nx in MESHES:
            row=lookup[('step',nx,method)]
            if row['status']!='selected':continue
            mesh,_,_=fixed_case(root,nx,'step')
            line=sample(mesh,selected_field(root,row),x,np.full_like(x,4.))
            ax.plot(x,line,linestyle=STYLES[nx],color=COLORS[method],label=f'{nx}×{MESHES[nx][1]}')
        ax.set(title=NAMES[method],xlabel='x（mm）',ylabel='E（MPa）');ax.legend(frameon=False)
    save_figure(fig,root,'step_profiles')

    for kind in ('step','smooth'):
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained',sharex=True,sharey='row')
        fig.suptitle(('阶梯场' if kind=='step' else '平滑场')+'：各网格独立L-curve选参',fontsize=12)
        for row_id,method in enumerate(('h1','exact_tv')):
            for col,nx in enumerate(MESHES):
                ax=axes[row_id,col]
                directory=root/f'{nx}x{MESHES[nx][1]}/{kind}_{method}'
                scans=read_json(directory/'scan.json');choice=read_json(directory/'selection.json')
                eligible=np.array(choice['eligible']);rho=np.array([s['rho'] for s in scans]);eta=np.array([s['eta'] for s in scans])
                ax.loglog(rho,np.where(eligible,eta,np.nan),'.-',color=COLORS[method],label='合格且范数可分辨')
                excluded=(~eligible)&(eta>0)
                ax.loglog(rho[excluded],eta[excluded],'x',color='gray',label='未参与曲率计算')
                if choice['optimal_index'] is not None:
                    i=choice['optimal_index'];ax.loglog(rho[i],eta[i],'o',markerfacecolor='none',markeredgecolor='black',markersize=10,label='选中点')
                ax.set(title=f'{NAMES[method]} / {nx}×{MESHES[nx][1]}',xlabel=r'$\|W_A^{1/2}r\|$',
                       ylabel=r'$\sqrt{\int |\nabla E|^2dA}$' if method=='h1' else r'$\int |\nabla E|dA$')
                ax.legend(frameon=False,fontsize=8)
        save_figure(fig,root,f'lcurve_{kind}')

    table=''
    for row in rows:
        val=[v for v in row.get('validation',[]) if v['converged']]
        if row['status']=='selected':
            interval=(f"{100*min(v['area_relative_rms'] for v in val):.2f}%—{100*max(v['area_relative_rms'] for v in val):.2f}%"
                      if len(val)==2 else '重复验证未全部合格')
            gamma=f"{row['selected_gamma']:.3g}";cost=str(row['evaluations'])
            width=row.get('midline_transition_width_mm')
            width=width_label(width)
        else:
            interval=row['status'];gamma=cost=width='—'
        table+=f"<tr><td>{'阶梯' if row['kind']=='step' else '平滑'}</td><td>{row['nx']}×{row['ny']}</td><td>{NAMES[row['method']]}</td><td>{gamma}</td><td>{interval}</td><td>{width}</td><td>{cost}</td><td>{row['converged_points']}/50</td></tr>"
    bounds=representation_bounds(root)
    floor_table=''.join(f"<tr><td>{nx}×{MESHES[nx][1]}</td><td>{100*next(b['dense_best_L2_relative_rms'] for b in bounds if b['nx']==nx and b['kind']=='step'):.2f}%</td></tr>" for nx in MESHES)
    complete=sum(r['status']=='selected' and len([v for v in r.get('validation',[]) if v['converged']])==2 for r in rows)
    step_ranges=[]
    for nx in MESHES:
        values=[100*v['area_relative_rms'] for v in lookup[('step',nx,'exact_tv')].get('validation',[]) if v['converged']]
        step_ranges.append(f'{min(values):.2f}%—{max(values):.2f}%' if len(values)==2 else '未完成合格验证')
    conclusion=('TV阶梯场随12×4 → 24×8 → 48×16加密，固定γ后的新噪声样本误差依次为'
        +' → '.join(step_ranges)+'。平滑场仍以梯度平方正则更好。'
        +f'{complete}/12组完成选参且两个新噪声样本均通过停止检查。真实DIC未参与，正式实验配置未修改。')
    extra_links=''.join(f' · <a href="{name}">{label}</a>' for name,label in [
        ('mesh_gradient_checks.json','加密梯度核验'),('precision_checks.json','收紧容差复核'),
        ('integration_sensitivity.json','积分精度复核')]
        if (root/name).exists())
    fixed_table=''.join(f"<tr><td>{r['nx']}×{MESHES[r['nx']][1]}</td><td>{100*r['area_relative_rms']:.2f}%</td><td>{width_label(r.get('midline_transition_width_mm'))}</td></tr>"
                       for r in fixed if r['kind']=='step' and r['method']=='exact_tv' and r['converged'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定观测的网格加密对照</title>
<style>body{{font:16px/1.7 'Microsoft YaHei',sans-serif;max-width:1240px;margin:30px auto;padding:0 24px;color:#263238}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}.result{{background:#fff3df;padding:15px}}a{{color:#0072b2}}</style>
<h1>增加单元数：同一观测下的二维自由模量对照</h1><p class="result">{conclusion}</p>
<p>固定30×8 mm域、48×16真值正演网格、833个观测位置、面积权重、载荷及每个样本的同一份噪声。只改变反演网格：12×4、24×8、48×16，分别65、225、833个独立节点E，不绑定不同y行。各网格使用同一均匀E初值；噪声0.2 μm是合成设定。</p>
<p>两种方法的位移插值及测试函数均为标准Q4形函数Nᵢ。模量均为Eₕ=ΣNᵢEᵢ，Eᵢ&gt;0。梯度平方目标为½rᵀWₐr+½γ∫(Eₓ²+Eᵧ²)dA；未平滑TV目标为½rᵀWₐr+γ∫√(Eₓ²+Eᵧ²)dA。没有SUPG、bubble-PG或预知分层参数。γ单位不同，分别扫描50点并按L-curve选择。</p>
<img src="mesh_error.png"><h2>固定γ后的两个新噪声样本</h2>
<p>误差在同一833点网格按面积评价。过渡宽度取选参样本中线从2200降至1600 MPa的距离（总反差的20%—80%）；只在两水平各有唯一交点时给出，否则记为—，不隐藏额外振荡。评估次数为该组选中点。12×4合格旧点复用，原预算失败点按本次统一上限重算。</p>
<table><tr><th>真值</th><th>网格</th><th>正则</th><th>γ</th><th>模量相对RMS</th><th>过渡宽度/mm</th><th>评估次数</th><th>合格扫描点</th></tr>{table}</table>
<img src="step_fields.png"><p>白虚线是真界面，仅供图上评价；没有进入反演目标或参数化。上述云图来自选参样本，误差范围来自另外两个固定γ的样本。</p><img src="step_profiles.png">
<h2>各档L-curve及停止状态</h2><p>灰叉包括预算内未达到停止条件的点和数值近常数场。它们均未参加曲率选择；没有改用真值误差最小点。</p><img src="lcurve_step.png"><img src="lcurve_smooth.png">
<h2>排除重新选γ的影响</h2><p>补充固定12×4所选TV权重γ={lookup[('step',12,'exact_tv')]['selected_gamma']:.4g}，直接提取三档扫描中相同γ且通过停止检查的场，不新增优化或按真值选择候选。下表是同一选参样本的误差与过渡宽度，可区分加密本身与重选γ的影响。</p><table><tr><th>网格</th><th>同γ阶梯误差</th><th>过渡宽度/mm</th></tr>{fixed_table}</table>
<h2>网格表达本身的限制</h2><p>下表直接把已知真值投影到各档连续Q4模量空间，计算同一96×32细积分网格上的最佳L²逼近误差。它只用于分离网格表达限制，不是位移反演结果，不参与γ或初值选择。</p>
<table><tr><th>网格</th><th>阶梯场最佳逼近误差</th></tr>{floor_table}</table>
<p>TV外迭代上限统一为1200，目标评估预算仍为12000次，所有收敛阈值保持原值。初始48×16试算在300步未通过约束残差，因此在正式扫描前提高外层上限；原试算记录保留。未通过一阶检查的点不参加选参，没有根据真值误差选γ。进程并行时机器负载不同，耗时保留为原始记录，效率比较优先使用目标评估次数。</p>
<p>另存公共细积分点误差，检查结果是否只在观测节点改善。最高反演网格与生成网格相同，但生成模量按高斯点解析突变、反演模量为连续Q4，两者仍不相同；此轮尚未排除真值正演网格自身的离散误差。没有验证实测噪声、实验ν、内部侧边界或真实DIC恢复。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="summary.json">逐组结果与细积分误差</a> · <a href="fixed_gamma_metrics.json">固定γ对照</a> · <a href="validation_metrics.csv">逐样本CSV</a> · <a href="representation_bounds.json">表达误差诊断</a>{extra_links}</p></html>'''
    (root/'index.html').write_text(page,encoding='utf-8')
    print(root/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    render(args.root.resolve())
