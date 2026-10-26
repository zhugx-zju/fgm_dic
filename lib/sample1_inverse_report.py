"""Audit and compare sample_1's two independent maximum-platform inversions.

This is post-processing. Direction hypotheses affect only independent
calibration forward predictions and regional comparison, never the inverse.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import partition_modulus, solve_forward
from .dic_case_workflow import load_prepared, _metrics, _regional_comparison
from .experiment_data import file_sha256, load_dic_frame_series
from .research_reporting import _json_value, modulus_width_profile


def _temporal_check(cfg, metadata):
    item = cfg.PLATFORMS['380N']
    series = load_dic_frame_series(Path(item['folder']), *item['stages'])
    for original, frozen in zip(series.metadata['sources'], metadata['platforms']['380N']['sources']):
        if original['sha256'] != frozen['sha256']:
            raise ValueError('Raw DIC changed after preparation')
    xy = series.coordinates @ np.asarray(cfg.REGISTRATION['matrix']).T
    uv = series.frame_displacements @ np.asarray(cfg.REGISTRATION['matrix']).T
    design = np.column_stack([np.ones(len(xy)), xy-xy.mean(axis=0)])
    coefficients = np.stack([np.linalg.lstsq(design, frame, rcond=None)[0] for frame in uv])
    axial_strain = coefficients[:, 1, 0]
    # Affine fits describe temporal variation only; none is removed from data.
    return dict(affine_axial_strain_by_frame=axial_strain.tolist(),
        affine_axial_strain_mean=float(axial_strain.mean()),
        affine_axial_strain_range=float(np.ptp(axial_strain)),
        affine_axial_strain_range_percent_of_mean=float(100*np.ptp(axial_strain)/abs(axial_strain.mean())),
        role='temporal consistency diagnostic; affine fields are not removed from observations')


def _analyze_run(root: Path, run: int, destination: Path):
    folder = root/f'results/dic_full/sample1_run{run}_380n_h1_20261026'
    prepared = folder/'01_prepared'
    cfg, mesh, cases, metadata, provenance = load_prepared(prepared)
    inverse = json.loads((folder/'03_inverse/run_metadata.json').read_text(encoding='utf-8'))
    scan = pd.read_csv(folder/'03_inverse/selection.csv')
    record = dict(run=run, output=str(folder), status=inverse['status'],
        selected_gamma=inverse.get('selected_gamma'),
        l_curve_message=inverse['l_curve_message'],scan_recovery=inverse.get('scan_recovery'),
        scanned_points=len(scan), solver_success_points=int(scan.converged.sum()),
        gtol_points=int(scan.gradient_tolerance_satisfied.sum()),
        total_scan_fe=int(scan.evaluations.sum()),
        mesh_elements=[mesh.nel_x,mesh.nel_y], parameters=mesh.n_nod,
        observations=len(cases[0].observation.x_meas),
        physical_model_mm=metadata['physical_model_mm'],
        force_assumed_N=metadata['force_pairing']['380N']['mean_corrected_force_N'],
        mark_registration=metadata['material_registration'],
        initialization=inverse['initialization'], provenance=provenance,
        requested_configuration_sha256=metadata['requested_configuration_sha256'],
        raw_dic_sources=metadata['platforms']['380N']['sources'],
        calibration_source_sha256=metadata['calibration_source_sha256'],
        temporal_check=_temporal_check(cfg,metadata))
    record['display_is_final'] = record['selected_gamma'] is not None
    before_path = folder/'03_inverse/selection_before_recovery.csv'
    if before_path.exists():
        before = pd.read_csv(before_path)
        record['initial_scan_success_points'] = int(before.converged.sum())
        record['recovery_extra_fe'] = int(scan.evaluations.sum()-before.evaluations.sum())
        if np.any(scan.evaluations > cfg.MAX_FUN) or np.any(scan.iterations > cfg.MAX_ITER):
            raise ValueError('Recovery exceeded an original point budget')
    if record['display_is_final']:
        record['display_gamma'] = record['selected_gamma']
        location_base = folder/'03_inverse'
        nus = cfg.NU_VALUES
    else:
        # Only display an existing algorithmic candidate; do not select gamma.
        if not np.isfinite(scan.curvature).any():
            return record, None
        peak = int(scan.curvature.idxmax())
        record['display_gamma'] = float(scan.loc[peak,'gamma'])
        record['display_role'] = 'unselected curvature-peak field; diagnostic only'
        record['corner_neighborhood'] = scan.iloc[max(0,peak-2):peak+3].replace({np.nan:None}).to_dict(orient='records')
        location_base = folder/'03_inverse/l_curve'/f"gamma_{record['display_gamma']:g}"
        nus = (cfg.PRIMARY_NU,)
    calibration = pd.DataFrame(metadata['calibration'])
    rows, regions, fields, final_checks, profiles = [], [], {}, [], []
    case = cases[0]
    for nu in nus:
        location = location_base/f'nu_{nu:.2f}' if record['display_is_final'] else location_base
        saved = json.loads((location/'run_metadata.json').read_text(encoding='utf-8'))
        with np.load(location/'fields.npz') as data:
            field = data['E_inverse'].copy()
            stored_prediction = data['380N_prediction'].copy() if '380N_prediction' in data else None
            stored_gradient = data['final_gradient'].copy()
        if not np.isfinite(field).all() or np.any(field <= 0):
            raise ValueError('Invalid saved modulus')
        u, _ = solve_forward(mesh,case.bc_info['tensile_end_force'],cfg.THICKNESS,nu,field)
        state = evaluate_observation(case,u)
        if stored_prediction is not None:
            np.testing.assert_allclose(state['prediction'],stored_prediction,atol=1e-11,rtol=0)
        metrics = _metrics(state)
        if record['display_is_final']:
            saved_metrics = pd.read_csv(folder/'03_inverse/comparison.csv')
            expected = saved_metrics[np.isclose(saved_metrics.nu,nu)].iloc[0]
            for key,value in metrics.items():
                np.testing.assert_allclose(value,expected[key],rtol=1e-10,atol=1e-12)
        # Re-evaluate the stored objective and adjoint, without optimization.
        from fgm_asm.material import MaterialInfo
        from fgm_asm.inverse_solver import _evaluate_inverse_state
        mesh.assemble_mass_matrix();mesh.get_regularization_matrix()
        objective = _evaluate_inverse_state(mesh,case.bc_info,None,MaterialInfo(nu=nu),
            record['display_gamma'],field,1,observation_cases=[case])
        np.testing.assert_allclose(objective['objective'],saved['solver']['final_cost'],rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(objective['grad_m'],stored_gradient,rtol=1e-5,atol=1e-10)
        final_checks.append(dict(nu=nu,prediction_recomputed=True,
            objective_recomputed=float(objective['objective']),
            gradient_inf_recomputed=float(np.max(np.abs(objective['grad_m']))),
            gradient_inf_saved=saved['solver']['final_gradient_inf'],
            converged=saved['solver']['converged'],stop_message=saved['solver']['message'],
            evaluations=saved['solver']['n_evaluations']))
        role='selected_gamma_result' if record['display_is_final'] else 'unselected_diagnostic'
        rows.append(dict(run=run,nu=nu,role=role,model='inverse' if record['display_is_final'] else 'inverse_diagnostic',direction='free_2d',**metrics))
        for direction,table in (('A_decreasing_raw_Y',calibration),
                                ('B_increasing_raw_Y',calibration.iloc[::-1].reset_index(drop=True))):
            reference = partition_modulus(table.mean_E_MPa,cfg.REGION_BOUNDARIES_X)
            forward_u,_ = solve_forward(mesh,case.bc_info['tensile_end_force'],cfg.THICKNESS,nu,reference)
            forward = evaluate_observation(case,forward_u)
            rows.append(dict(run=run,nu=nu,role='independent_calibration_forward',model='calibration_forward',direction=direction,**_metrics(forward)))
            comparison = _regional_comparison(mesh,field,table,cfg.REGION_BOUNDARIES_X)
            comparison.insert(0,'direction',direction)
            comparison.insert(0,'nu',nu)
            comparison.insert(0,'run',run)
            comparison.insert(0,'inverse_field_role',role)
            regions.extend(comparison.to_dict(orient='records'))
            if nu == cfg.PRIMARY_NU:
                fields[f'{run}_{direction}_prediction'] = forward['prediction']
        if nu == cfg.PRIMARY_NU:
            ref = partition_modulus(calibration.mean_E_MPa,cfg.REGION_BOUNDARIES_X)
            profile = modulus_width_profile(mesh,field,ref)
            profile.insert(0,'run',run)
            profiles.append(profile)
            fields.update({f'{run}_E':field,f'{run}_coordinates':mesh.coord,
                           f'{run}_inverse_prediction':state['prediction']})
            record['modulus_nodal_range_MPa'] = [float(field.min()),float(field.max())]
            record['width_mean_endpoints_MPa'] = profile.inverse_width_mean_E_MPa.iloc[[0,-1]].tolist()
            record['width_mean_range_MPa'] = [float(profile.inverse_width_mean_E_MPa.min()),
                                             float(profile.inverse_width_mean_E_MPa.max())]
    record.update(final_checks=final_checks,
        calibration=calibration.to_dict(orient='records'),
        region_boundaries_x_mm=list(cfg.REGION_BOUNDARIES_X))
    return record,dict(metrics=pd.DataFrame(rows),regions=pd.DataFrame(regions),
                      fields=fields,profiles=pd.concat(profiles,ignore_index=True),mesh=mesh)


def build_report(root: Path):
    output = root/'results/dic_full/sample1_380n_20261026'
    output.mkdir(parents=True,exist_ok=True)
    records, data = [], []
    for run in (2,3):
        record,item = _analyze_run(root,run,output)
        records.append(record)
        if item is not None:data.append(item)
    table_html = ''
    text = ['# sample_1 两次最高平台自由二维模量反演', '',
        '第1次弃用；第2、3次分别用自身DIC初始化、扫描50个γ并计算独立二维Q4模量。'
        '第2次选参及四个ν的最终反演完成；第3次选参未通过，展示场仅为曲率峰候选诊断，不是最终识别结论。'
        '两组均采用原始面积加权双分量位移失配与H1正则，不绑定不同y行，不输入层数或界面作为反演先验。', '',
        '载荷按用户授权，分别假定为380−46.3272=333.6728 N和380−54.6128=325.3872 N。'
        '最高平台无同步实测力；早于试验机启动的DIC参考态与初始力状态的等价仍是假定。'
        '第3次按确认的5 Hz仍不能与力记录以单一时间差对齐，不能称为已验证的实验模量。', '',
        '两条划线约距220 mm试样外端各53 mm，名义间距114 mm；ROI约在两条线之间。'
        '只按中点定位，保留两组实际毫米跨度，不补点、不缩放到114 mm。'
        '填充率方向未确认：A表示沿原始+Y为100→40%，B为40→100%。'
        '以下双方向比较发生在反演以后；方向对照不是独立制造位置测量。', '',
        '| 记录 | 测点 | 单元 | 独立E参数 | 求解器停止成功点/50 | GTOL点/50 | 所选γ |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in records:
        text.append(f"| {r['run']} | {r['observations']} | {r['mesh_elements'][0]}×{r['mesh_elements'][1]} | "
                    f"{r['parameters']} | {r['solver_success_points']} | {r['gtol_points']} | {r['selected_gamma']} |")
    text += ['', '原扫描成功点分别为38/50、43/50。使用原有--resume仅恢复线搜索异常，'
             '第2次2点恢复，第3次1点仍失败；各γ累计迭代/FE预算未增加，未恢复已耗尽迭代预算的弱γ。', '']
    for r in records:
        if not r['display_is_final']:
            text += [f"记录{r['run']}的曲率峰候选γ={r['display_gamma']:.12g}未被选定。"
                     '其邻域状态如下；所展示二维场仅用于诊断，不作为最终γ选择或正式识别。', '',
                     '| γ | 求解器成功 | 迭代数 | 停止原因 |','|---|---|---:|---|']
            for row in r['corner_neighborhood']:
                text.append(f"| {row['gamma']:.12g} | {row['converged']} | {row['iterations']} | {row['stop_message']} |")
    text += ['', '只有求解器状态通过的点参与L-curve；无有效拐点则停止，不用标定误差补选γ。'
             '第2次在ν=0.26下独立选γ后，固定本组γ比较ν=0.24/0.26/0.28/0.33；第3次未进入最终ν反演和第4步。'
             '函数下降停止与GTOL达标分别报告。', '']
    if len(data)==2:
        metrics = pd.concat([d['metrics'] for d in data],ignore_index=True)
        regions = pd.concat([d['regions'] for d in data],ignore_index=True)
        profiles = pd.concat([d['profiles'] for d in data],ignore_index=True)
        metrics.to_csv(output/'direction_displacement_metrics.csv',index=False)
        regions.to_csv(output/'direction_regional_comparison.csv',index=False)
        profiles.to_csv(output/'width_profiles.csv',index=False)
        np.savez_compressed(output/'direction_fields.npz',**{k:v for d in data for k,v in d['fields'].items()})
        summary_rows=[]
        for r in records:
            for direction in ('A_decreasing_raw_Y','B_increasing_raw_Y'):
                a=regions[(regions.run==r['run'])&(regions.nu==.26)&(regions.direction==direction)]
                b=metrics[(metrics.run==r['run'])&(metrics.nu==.26)&(metrics.direction==direction)].iloc[0]
                summary_rows.append(dict(run=r['run'],direction=direction,inverse_is_final=r['display_is_final'],
                    mean_absolute_regional_difference_percent=float(a.difference_percent.abs().mean()),
                    maximum_absolute_regional_difference_percent=float(a.difference_percent.abs().max()),
                    calibrated_rmse_ux_um=float(b.rmse_ux_mm*1000),calibrated_rmse_uy_um=float(b.rmse_uy_mm*1000)))
        summary=pd.DataFrame(summary_rows)
        summary.to_csv(output/'direction_summary.csv',index=False)
        table_html = ('<h2>方向对照（ν=0.26）</h2>'+summary.to_html(index=False,border=0)+
                      '<h2>位移误差（mm；ν=0.26）</h2>'+metrics[np.isclose(metrics.nu,.26)].to_html(index=False,border=0))
        text += ['## 主ν=0.26的结果与方向对照', '',
            '| 记录 | 模型/方向 | 轴向RMSE / μm | 横向RMSE / μm | 七区均值平均绝对标定差异 / % |',
            '|---|---|---:|---:|---:|']
        for r in records:
            for row in metrics[(metrics.run==r['run'])&(metrics.nu==.26)].to_dict(orient='records'):
                match=summary[(summary.run==r['run'])&(summary.direction==row['direction'])]
                error='—' if match.empty else f"{match.iloc[0].mean_absolute_regional_difference_percent:.3f}"
                text.append(f"| {r['run']} | {row['model']} / {row['direction']} | {row['rmse_ux_mm']*1000:.3f} | "
                            f"{row['rmse_uy_mm']*1000:.3f} | {error} |")
        text.append('')
        for r in records:
            subset=summary[summary.run==r['run']]
            by_displacement=subset.loc[subset.calibrated_rmse_ux_um.idxmin(),'direction']
            by_modulus=subset.loc[subset.mean_absolute_regional_difference_percent.idxmin(),'direction']
            r['direction_preference_axial_response']=by_displacement
            r['direction_preference_regional_modulus']=by_modulus
            preference=by_displacement if by_displacement==by_modulus else 'inconsistent_between_metrics'
            r['direction_response_preference']=preference
            text.append(f"记录{r['run']}：轴向标定正演误差支持{by_displacement}，"
                        f"区域模量差异支持{by_modulus}。"
                        '这两项来自同一组响应数据，不能视为两次独立制造方向验证。')
        text.append('')
        with plt.rc_context({'font.family':'Microsoft YaHei','font.size':10,
                             'axes.unicode_minus':False,'svg.fonttype':'path'}):
            fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
            for ax,r,d in zip(axes,records,data):
                profile=d['profiles'];x=profile.x_mm.to_numpy()
                mean=profile.inverse_width_mean_E_MPa.to_numpy();spread=profile.inverse_transverse_sd_MPa.to_numpy()
                ax.plot(x,mean,color='#0072B2',lw=1.8,label='自由二维反演：宽度平均')
                ax.fill_between(x,mean-spread,mean+spread,color='#0072B2',alpha=.17,label='横向空间标准差')
                cal=pd.DataFrame(r['calibration']);dense=np.linspace(0,r['physical_model_mm'][0],1601)
                for values,color,label in ((cal.mean_E_MPa,'#333333','标定A：原始+Y方向100→40%'),
                    (cal.mean_E_MPa.iloc[::-1],'#D55E00','标定B：原始+Y方向40→100%')):
                    reference=partition_modulus(values,r['region_boundaries_x_mm'])
                    ax.plot(dense,reference(dense,dense*0),color=color,lw=1.15,ls='--',label=label)
                ax.set(ylabel='E (MPa)',title=f"记录{r['run']}，ν=0.26，假定净力 {r['force_assumed_N']:.4f} N"+
                       ('；γ已选定' if r['display_is_final'] else '；未选参，仅曲率峰候选诊断'))
                ax.legend(frameon=False,fontsize=9,ncol=2)
                ax.spines[['top','right']].set_visible(False)
            axes[-1].set_xlabel('FEM x = 原始 Y − ROI内最小参考Y (mm)')
            fig.savefig(output/'width_profile_directions.png',dpi=300)
            fig.savefig(output/'width_profile_directions.svg')
            plt.close(fig)
            fig,axes=plt.subplots(2,1,figsize=(11,5.3),layout='constrained')
            vmin=min(r['modulus_nodal_range_MPa'][0] for r in records)
            vmax=max(r['modulus_nodal_range_MPa'][1] for r in records)
            for ax,r,d in zip(axes,records,data):
                mesh=d['mesh'];field=d['fields'][f"{r['run']}_E"].reshape(mesh.nods_y,mesh.nods_x)
                im=ax.pcolormesh(mesh.plot_x,mesh.plot_y,field,shading='auto',cmap='viridis',vmin=vmin,vmax=vmax)
                ax.set(xlabel='FEM x：沿原始 +Y (mm)',ylabel='FEM y (mm)',aspect='equal',
                       title=f"记录{r['run']}：自由二维模量，ν=0.26"+
                       ('；γ已选定' if r['display_is_final'] else '；未选参，仅曲率峰候选诊断'))
                fig.colorbar(im,ax=ax,label='E (MPa)',shrink=.85)
            fig.savefig(output/'modulus_two_runs.png',dpi=300)
            fig.savefig(output/'modulus_two_runs.svg')
            plt.close(fig)
        text += ['', '![两次二维模量](../results/dic_full/sample1_380n_20261026/modulus_two_runs.png)', '',
                 '![两种方向对照](../results/dic_full/sample1_380n_20261026/width_profile_directions.png)', '']
        for r in records:
            text += [f"记录{r['run']}（{'已选参结果' if r['display_is_final'] else '未选参诊断场'}）：节点E范围{r['modulus_nodal_range_MPa'][0]:.3f}—"
                     f"{r['modulus_nodal_range_MPa'][1]:.3f} MPa；完整场、宽度平均及横向离散需合并评价。"
                     f"候选15帧全域仿射轴向应变的范围/均值为"
                     f"{r['temporal_check']['affine_axial_strain_range_percent_of_mean']:.3f}%。"
                     '仿射量仅用于时间变化诊断，未从反演位移中扣除。', '']
        text += ['## 停止与核验', '']
        for r in records:
            text.append(f"记录{r['run']}扫描共{r['total_scan_fe']}次FE函数计算；初值"
                        f"{r['initialization']['uniform_E_MPa']:.6f} MPa，仅由本组DIC估计。"
                        f"恢复额外使用{r.get('recovery_extra_fe',0)}次FE，包含在各点原预算内。")
            for check in r['final_checks']:
                text.append(f"- ν={check['nu']:.2f}：{check['stop_message']}；"
                            f"重算梯度∞范数={check['gradient_inf_recomputed']:.6g}。")
            text.append('')
    text += ['## 适用范围与复现', '',
        '此次仅使用两次最高平台，各次分别识别；两次来自同一试样的不同加载记录，不作为独立试样重复。'
        '不依据标定选择γ或ν，不把ν敏感性中误差最小者解释为实测ν。'
        '方向比较只评价本数据和近似划线定位条件下的相容性。'
        '独立标定沿用mixed_infill_20261017的independent口径，保留各区原始组均值、'
        '试样标准差、样本数和批次；跨批桥接及向本试样转移的限制仍保留，未通过调整标定降低误差。'
        '两组载荷、较早DIC参考态、毫米坐标标定、横向自由侧边近似及截取边界等效仍有物理验证限制。'
        '尤其第3次时序不一致不能由模量反演修复。优化停止或位移拟合改善不等于实验物理验证通过。', '',
        '运行入口为dic_sample1_run2_380n_config.py、dic_sample1_run3_380n_config.py；'
        '调用原script.step1_prepare_dic至step4_compare_results，均带对应--config。'
        '第2次完成四步，第3次止于第3步，未调用第4步生成最终对照。'
        '已有结果重绘使用--plot-only，不重新求解。复核与双方向后处理：', '',
        '```powershell', 'python -X utf8 -B -m lib.sample1_inverse_report', '```', '',
        '逐帧输入哈希、原始单位/参考坐标、力假定、配准和网格保存在各01_prepared；'
        '逐γ场与停止信息保存在03_inverse。小规模解析与有限差分检查保存在汇总目录input_audit。'
        '本报告生成时重新计算了保存E的FE回代、两分量误差、目标函数及伴随梯度；'
        '方向B独立正演复用同一FE内核，不调整标定模量。']
    (root/'docs/sample1_380n_inverse_20261026.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    result=dict(records=records,code_sha256=file_sha256(Path(__file__)),
        smoke_checks=json.loads((output/'input_audit/smoke_checks.json').read_text(encoding='utf-8')),
        inverse_direction_independent=True,force_assumed=True,direction_independently_verified=False)
    (output/'summary.json').write_text(json.dumps(_json_value(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (root/'docs/sample1_380n_inverse_20261026.json').write_text(
        json.dumps(_json_value(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    links=''.join(f'<li>记录{r["run"]}: <a href="../sample1_run{r["run"]}_380n_h1_20261026/结果总览.html">分步结果</a> · '+
        (f'<a href="../sample1_run{r["run"]}_380n_h1_20261026/04_comparison/打开查看结果.html">最终对照</a>' if r['display_is_final'] else
         f'<a href="../sample1_run{r["run"]}_380n_h1_20261026/03_inverse/云图.html">选参失败证据</a>')+'</li>' for r in records)
    body=''.join(f'<p>{html.escape(line)}</p>' for line in text if line and not line.startswith(('![','|','```')))
    page=f'''<!doctype html><meta charset="utf-8"><title>sample_1最高平台二维反演</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.65 Microsoft YaHei,sans-serif}}img{{width:100%}}a{{color:#0969a6}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px;border-bottom:1px solid #ddd}}</style>
<h1>sample_1：第2、3次最高平台自由二维模量反演</h1><ul>{links}</ul>
<p><strong>第2次完成选参及最终反演；第3次未通过选参，以下第3次场仅为已有曲率峰候选诊断。</strong></p>
<p>净力为用户授权假定；填充率方向未独立确认，第3次力—帧配对未通过。</p>
<img src="modulus_two_runs.png" alt="两次二维模量"><img src="width_profile_directions.png" alt="两种方向对照">{table_html}
{body}<p><a href="direction_summary.csv">方向简表</a> · <a href="summary.json">核验与来源</a></p>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    for record in records:
        case_root=Path(record['output'])
        for path in case_root.rglob('*.html'):
            content=path.read_text(encoding='utf-8')
            if 'sample1-conditional-notice' in content:
                continue
            link=Path(os.path.relpath(output/'index.html',path.parent)).as_posix()
            notice=(f'<aside id="sample1-conditional-notice" style="padding:12px;background:#fff2cc;'
                f'border:1px solid #d5b44c"><strong>本组条件：</strong>净力{record["force_assumed_N"]:.4f} N'
                '为假定值，无最高平台同步力；材料方向未确认。默认标定图仅为方向A，'
                '不作为已测得的制造位置。'+('第3次力—DIC时序未通过。' if record['run']==3 else '')+
                f'<a href="{html.escape(link)}">查看双方向对照与全部限制</a></aside>')
            if '<body>' in content:content=content.replace('<body>','<body>'+notice,1)
            else:content=content.replace('<h1>',notice+'<h1>',1)
            path.write_text(content,encoding='utf-8')
    print(output/'index.html')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    build_report(args.root.resolve())


if __name__ == '__main__':
    main()
