"""Evaluate frozen exact-TV weights on fresh noise and render saved results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz

from fgm_asm.exact_tv import solve_exact_tv
from lib.exact_tv_study import (DEFAULT_OUTPUT, dataset, field_metrics, save_json,
                                solve_h1, truth_function)


NAMES = {'h1': '梯度平方正则', 'exact_tv': '未平滑TV正则'}
COLORS = {'h1': '#0072B2', 'exact_tv': '#D55E00'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def evaluate(root):
    pilot = root/'selection_pilot'
    selections = read_json(pilot/'selections.json')
    if len(selections) != 4:
        raise ValueError('Complete both 50-point scans before evaluation')
    if not all(row['status'] == 'selected' for row in selections):
        save_json(root/'decision.json', dict(advance_to_DIC=False,
                  reason='A required L-curve selection failed.', selections=selections))
        return
    rules = read_json(pilot/'evaluation_gate.json')
    directory = root/'frozen_weight_validation'
    directory.mkdir(exist_ok=False)
    lookup = {(s['kind'], s['method']):s['selected_gamma'] for s in selections}
    rows = []
    for seed in rules['evaluation_seeds']:
        for kind in ('uniform', 'smooth', 'step'):
            mesh, case, initial, truth, meta = dataset(kind, seed=seed)
            obs = case.observation
            np.savez_compressed(directory/f'data_{kind}_{seed}.npz',
                coordinates=np.column_stack((obs.x_meas, obs.y_meas)),
                observed=obs.u_measured, weights=obs.weight_vector, initial=initial)
            save_json(directory/f'data_{kind}_{seed}.json', meta)
            for method in ('h1', 'exact_tv'):
                gamma = lookup[('step' if kind == 'uniform' else kind, method)]
                result = (solve_h1(mesh, case, initial, gamma) if method == 'h1' else
                          solve_exact_tv(mesh, [case], initial, gamma))
                row = dict(kind=kind, seed=seed, method=method, gamma=gamma,
                           converged=result['converged'], status=result['status'],
                           evaluations=result['evaluations'],
                           **field_metrics(mesh, case, result['E'], kind))
                rows.append(row)
                np.savez_compressed(directory/f'{kind}_{seed}_{method}.npz',
                                     E=result['E'], truth=truth, coordinates=mesh.coord)
                if method == 'exact_tv':
                    save_json(directory/f'{kind}_{seed}_{method}_history.json', result['history'])
                save_json(directory/'metrics.json', rows)
                print('FROZEN', row, flush=True)
    checks = []
    for seed in rules['evaluation_seeds']:
        for kind, bound in [('step', rules['step_TV_error_max_ratio_to_H1_each_seed']),
                            ('smooth', rules['smooth_TV_error_max_ratio_to_H1_each_seed'])]:
            values = {r['method']:r['area_relative_rms'] for r in rows
                      if r['seed'] == seed and r['kind'] == kind}
            ratio = values['exact_tv']/values['h1']
            checks.append(dict(seed=seed, kind=kind, TV_to_H1_error_ratio=ratio,
                               limit=bound, passed=ratio <= bound))
    uniform_passed = all(r['field_std_MPa'] <= rules['uniform_false_structure_std_max_MPa']
                         for r in rows if r['kind'] == 'uniform')
    converged = all(r['converged'] for r in rows)
    passed = uniform_passed and converged and all(c['passed'] for c in checks)
    save_json(root/'decision.json', dict(advance_to_DIC=bool(passed),
        convergence_passed=converged, uniform_structure_passed=uniform_passed,
        uniform_structure_by_method={method:all(
            r['field_std_MPa'] <= rules['uniform_false_structure_std_max_MPa']
            for r in rows if r['kind'] == 'uniform' and r['method'] == method)
            for method in ('h1','exact_tv')},
        comparison_checks=checks, actual_DIC_run=False,
        reason='Frozen-weight synthetic gate; no truth-based retuning.'))
    with (directory/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(root):
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False,
                         'font.size':10, 'svg.fonttype':'none'})
    pilot = root/'selection_pilot'
    selections = read_json(pilot/'selections.json')
    if len(selections) != 4 or any(s['status'] != 'selected' for s in selections):
        raise ValueError('Field comparison requires four valid selected fields')
    lookup = {(s['kind'], s['method']):s for s in selections}
    fields, interpolation = {}, {}
    for kind in ('smooth', 'step'):
        for method in ('h1', 'exact_tv'):
            directory = pilot/f'{kind}_{method}'
            index = read_json(directory/'selection.json')['optimal_index']
            with np.load(directory/'fields.npz') as arrays:
                fields[(kind, method)] = arrays['E'][index].copy()
        interpolation[kind] = load_npz(pilot/f'{kind}_H.npz')[::2,::2]
    x, y = np.linspace(0, 30, 49), np.linspace(0, 8, 17)
    xx, yy = np.meshgrid(x, y)
    fine = {}
    for kind in ('smooth','step'):
        fine[(kind,'truth')] = truth_function(kind)(xx, yy)
        for method in ('h1','exact_tv'):
            fine[(kind,method)] = np.asarray(interpolation[kind] @ fields[(kind,method)]).reshape(17,49)
    vmin = 100*np.floor(min(np.min(v) for v in fine.values())/100)
    vmax = 100*np.ceil(max(np.max(v) for v in fine.values())/100)
    fig, axes = plt.subplots(2,3,figsize=(13,5.5),layout='constrained')
    fig.suptitle('同一二维节点Q4模量空间；位移试函数与测试函数均为 $N_i$\n'
        r'$J_{H1}=\frac{1}{2}r^T W_A r+\frac{\gamma}{2}\int(E_x^2+E_y^2)dA$'
        '     '
        r'$J_{TV}=\frac{1}{2}r^T W_A r+\gamma\int\sqrt{E_x^2+E_y^2}dA$',fontsize=12)
    for row, kind in enumerate(('smooth','step')):
        for col, method in enumerate(('truth','h1','exact_tv')):
            ax = axes[row,col]
            im = ax.pcolormesh(x,y,fine[(kind,method)],shading='auto',cmap='viridis',vmin=vmin,vmax=vmax)
            ax.set_aspect('equal')
            if method == 'truth':
                title = '合成真值：'+('平滑二维场' if kind == 'smooth' else '斜界面阶梯场')
            else:
                s = lookup[(kind,method)]
                title = f"{NAMES[method]}：γ={s['selected_gamma']:.2e}\n面积加权模量相对RMS={100*s['metrics']['area_relative_rms']:.2f}%"
            ax.set_title(title,fontsize=10)
            ax.set_xlabel('x（mm）')
            ax.set_ylabel('y（mm）')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='杨氏模量 E（MPa）',shrink=.7)
    for ext in ('png','svg'):
        fig.savefig(root/f'field_comparison.{ext}',dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for ax, kind in zip(axes,('smooth','step')):
        if kind == 'step':
            location = .45*30.-.6*4.
            ax.plot([0,location,location,30],[2400,2400,1400,1400],
                    color='black',label='合成真值',linewidth=1.7)
        else:
            ax.plot(x,fine[(kind,'truth')][8],color='black',label='合成真值',linewidth=1.7)
        for method, style in [('h1','--'),('exact_tv','-')]:
            ax.plot(x,fine[(kind,method)][8],style,color=COLORS[method],label=NAMES[method],linewidth=1.6)
        ax.set(title=('平滑场' if kind == 'smooth' else '斜界面阶梯场')+'：中线 y=4 mm',xlabel='x（mm）',ylabel='E（MPa）')
        ax.legend(frameon=False)
    for ext in ('png','svg'):
        fig.savefig(root/f'centerline_comparison.{ext}',dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row, kind in enumerate(('smooth','step')):
        for col, method in enumerate(('h1','exact_tv')):
            ax = axes[row,col]
            directory = pilot/f'{kind}_{method}'
            scan = read_json(directory/'scan.json')
            chosen = read_json(directory/'selection.json')
            eligible = np.array(chosen['eligible'])
            rho = np.array([s['rho'] for s in scan])
            eta = np.array([s['eta'] for s in scan])
            ax.loglog(rho[eligible],eta[eligible],'.-',color=COLORS[method],label='通过停止检查且范数可分辨')
            bad = ~eligible & (eta > 0)
            ax.loglog(rho[bad],eta[bad],'x',color='gray',label='未纳入曲率计算')
            i=chosen['optimal_index']
            ax.loglog(rho[i],eta[i],'o',markerfacecolor='none',markeredgecolor='black',markersize=10,label='L-curve选中点')
            ax.set(title=('平滑场' if kind == 'smooth' else '阶梯场')+' / '+NAMES[method],
                   xlabel=r'$\|W_A^{1/2}r\|$',
                   ylabel=(r'$\sqrt{\int |\nabla E|^2 dA}$' if method == 'h1' else r'$\int |\nabla E| dA$'))
            ax.legend(frameon=False,fontsize=8)
    for ext in ('png','svg'):
        fig.savefig(root/f'l_curves.{ext}',dpi=220)
    plt.close(fig)
    decision=read_json(root/'decision.json')
    rows=read_json(root/'frozen_weight_validation/metrics.json')
    table=''
    for kind in ('smooth','step','uniform'):
        for method in ('h1','exact_tv'):
            values=[100*r['area_relative_rms'] for r in rows if r['kind']==kind and r['method']==method]
            source='step' if kind=='uniform' else kind
            sel=lookup[(source,method)]
            table+=f"<tr><td>{dict(smooth='平滑场',step='斜界面阶梯场',uniform='均匀场')[kind]}</td><td>{NAMES[method]}</td><td>{sel['selected_gamma']:.4g}</td><td>{min(values):.2f}%—{max(values):.2f}%</td><td>{sel['evaluations']}</td></tr>"
    conclusion=('通过预设合成筛选，仍需更大网格核验。' if decision['advance_to_DIC'] else
                '未通过预设合成筛选；没有运行真实DIC，没有替换现有主算法。')
    verification_link = (' · <a href="verification/derivative_checks.json">梯度与解析解核验</a>'
                         if (root/'verification/derivative_checks.json').exists() else '')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>未平滑TV：二维自由模量验证</title>
<style>body{{font:16px/1.7 'Microsoft YaHei',sans-serif;max-width:1200px;margin:30px auto;padding:0 24px;color:#263238}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}code{{background:#f3f5f6}}.result{{padding:14px;background:#fff3df}}a{{color:#0072b2}}</style>
<h1>未平滑TV与梯度平方正则：受控合成验证</h1><p class="result">{conclusion}</p>
<p>两种方法均采用连续Q4插值 E<sub>h</sub>=ΣN<sub>i</sub>E<sub>i</sub>，每个节点E独立，不绑定不同y行。位移插值与测试函数均为N<sub>i</sub>。本轮没有使用SUPG或bubble-PG。</p>
<p>梯度平方正则：J=½rᵀW<sub>A</sub>r+½γ∫(E<sub>x</sub>²+E<sub>y</sub>²)dA。未平滑TV：J=½rᵀW<sub>A</sub>r+γ∫√(E<sub>x</sub>²+E<sub>y</sub>²)dA。TV不含Huber阈值或根号内ε；用高斯点离散积分。两个γ单位不同，分别选参。</p>
<p>30×8 mm合成域；反演12×4单元、65个独立节点模量，正演48×16单元。噪声标准差0.2 μm是人为设定，不代表已测DIC噪声。ν=0.26；厚3.2 mm；牵引按真实承载宽20 mm换算。斜界面位置和模量值只用于造数据与误差评价，不进入反演。</p>
<p>各扫50个γ，仅按180N型合成识别数据的L-curve选参。TV检查约束残差、对偶可行性、Fenchel间隙及log(E)一阶驻点残差；内层目标下降不等于外层收敛。初值仅由同一识别位移拟合均匀E。每个候选点预算12000次目标评估（SciPy线搜索可能略超），实际次数单独保存。</p>
<img src="field_comparison.png"><img src="centerline_comparison.png">
<h2>冻结γ后更换两个噪声样本</h2><p>表中误差为种子20261023、20261024的范围；γ由种子20261022确定后固定。均匀场直接使用阶梯场选出的γ。评估次数列为首组选中点的前向/伴随目标评估次数。</p>
<table><tr><th>真值</th><th>具体正则</th><th>固定γ</th><th>模量面积加权相对RMS</th><th>选中点评估次数</th></tr>{table}</table>
<img src="l_curves.png">
<p>进入真实DIC前的筛选条件预先规定：两类非均匀场均选到受合格求解支持的拐点；更换噪声后均收敛；阶梯误差≤H1的80%，平滑误差≤H1的120%；两种方法的均匀场模量标准差均≤9 MPa。这是本次筛选规则，不是TV的一般理论界限。TV在均匀场的标准差仅约6—8×10⁻⁶ MPa，未出现可分辨伪结构；均匀场联合条件失败来自H1的一个噪声样本，不能归因于TV。</p>
<p>连续Q4模量仍只能在有限网格尺度上表示突变过渡；本试验不能证明真实DIC可识别任意细界面，也没有验证泊松比、内部侧边界等实验物理近似。ADMM处理非线性反演只给出局部一阶检查，不保证全局最优。</p>
<p><a href="selection_pilot/protocol.json">扫描协议</a> · <a href="selection_pilot/evaluation_gate.json">事先声明的筛选条件</a> · <a href="decision.json">筛选判定</a> · <a href="frozen_weight_validation/metrics.csv">逐次误差CSV</a>{verification_link}</p>
<p>原理参考：<a href="https://stanford.edu/~boyd/papers/admm_distr_stats.html">Boyd等，ADMM综述（2011）</a>；其凸问题收敛结论不直接适用于本非线性反演。</p></html>'''
    (root/'index.html').write_text(page,encoding='utf-8')
    print(root/'index.html',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['evaluate','render'])
    parser.add_argument('--root',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    (evaluate if args.action=='evaluate' else render)(args.root.resolve())
