"""Conditional shared-E experiment using three independent DIC references.

This limited diagnostic calls the existing inverse kernel. It does not replace
the four-step workflow, perform gamma selection, or certify registration.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm import MeshInfo
from fgm_asm.material import MaterialInfo
from fgm_asm.research import make_observation_case, solve_forward
from fgm_asm.load_case import evaluate_observation
from fgm_asm.inverse_solver import _evaluate_inverse_state
from .dic_case_workflow import load_prepared, homogeneous_dic_initial, _optimize
from .dic_preprocessing import relative_displacement_errors
from .dic_gamma_stability import GAMMAS
from .dic_calibration_curve_comparison import metrics as calibration_metrics, width_profile
from .experiment_data import file_sha256


def build_common_cases(roots):
    loaded = [load_prepared(root/'01_prepared') for root in roots]
    length = min(item[0].GEO_L for item in loaded)
    width = min(item[0].GEO_H for item in loaded)
    cfg = SimpleNamespace(**vars(loaded[0][0]))
    cfg.GEO_L, cfg.GEO_H = length, width
    mesh = MeshInfo(length, width, 121, 19)
    cases, audit, arrays = [], [], {}
    for root, (original, _, original_cases, metadata, provenance) in zip(roots, loaded):
        for key in ('THICKNESS', 'PRIMARY_NU', 'REGULARIZATION_MODE', 'MAX_ITER', 'MAX_FUN', 'FTOL', 'GTOL'):
            if getattr(original, key) != getattr(cfg, key):
                raise ValueError(f'Confounded source setting {key}')
        if len(original_cases) != 1 or original.REGULARIZATION_MODE != 'gradient':
            raise ValueError('Expected one frozen H1 case per source')
        old = original_cases[0]
        xy = np.column_stack([old.observation.x_meas, old.observation.y_meas])
        # Retain the earlier conditional center placement; never rescale spans.
        shift = np.array([(length-original.GEO_L)/2, (width-original.GEO_H)/2])
        xy = xy+shift
        keep = ((xy[:, 0] >= 0) & (xy[:, 0] <= length) & (xy[:, 1] >= 0) & (xy[:, 1] <= width))
        uv = old.observation.u_measured.reshape(-1, 2)[keep].copy()
        area = old.observation.point_weights[keep].copy()
        stress = old.bc_info['tensile_end_force']/(original.GEO_H*original.THICKNESS)
        model_force = stress*width*cfg.THICKNESS
        case = make_observation_case(mesh, old.name, model_force, cfg.THICKNESS, xy[keep], uv,
            point_weights=area, metadata=dict(independent_reference=True, registration_verified=False,
                source_prepared=str((root/'01_prepared').resolve()), traction_MPa=stress))
        np.testing.assert_allclose(case.bc_info['tensile_end_force']/(width*cfg.THICKNESS), stress, rtol=1e-14)
        np.testing.assert_array_equal(case.observation.u_measured.reshape(-1, 2), uv)
        np.testing.assert_array_equal(case.observation.weight_vector.reshape(-1, 2), old.observation.weight_vector.reshape(-1, 2)[keep])
        cases.append(case)
        _, initial = homogeneous_dic_initial(mesh, [case], cfg.THICKNESS, cfg.PRIMARY_NU)
        audit.append(dict(case=old.name, source=str(root.resolve()), source_geometry_mm=[original.GEO_L, original.GEO_H],
            common_geometry_mm=[length, width], translation_mm=shift.tolist(), length_scale=1.,
            points_original=len(keep), points_kept=int(keep.sum()), force_common_N=model_force,
            original_force_N=old.bc_info['tensile_end_force'], traction_MPa=stress,
            homogeneous_dic_E_MPa=initial['uniform_E_MPa'], registration_verified=False,
            reference_policy='Each independent reference has its own three eliminated rigid modes.', **provenance))
        arrays.update({f'{old.name}_coordinates': xy[keep], f'{old.name}_observed': uv,
                       f'{old.name}_point_area_mm2': area})
    return cfg, mesh, cases, audit, arrays, loaded[0][3]['calibration']


def kernel_checks(cfg, mesh, cases, gamma):
    small = MeshInfo(mesh.geo_l, mesh.geo_h, 8, 3)
    observations = [make_observation_case(small, case.name, case.bc_info['tensile_end_force'], cfg.THICKNESS,
        np.column_stack([case.observation.x_meas, case.observation.y_meas])[::13],
        case.observation.u_measured.reshape(-1, 2)[::13], point_weights=case.observation.point_weights[::13]) for case in cases]
    field = 1250*np.exp(.1*np.sin(2*np.pi*small.coord[:, 0]/small.geo_l)*np.cos(np.pi*small.coord[:, 1]/small.geo_h))
    def state(field, selected, gamma):
        return _evaluate_inverse_state(small, None, None, MaterialInfo(cfg.PRIMARY_NU), gamma,
                                       field, 0, observation_cases=selected)
    combined = state(field, observations, 3*gamma)
    separate = [state(field, [case], gamma) for case in observations]
    np.testing.assert_allclose(combined['objective'], sum(row['objective'] for row in separate), rtol=1e-12)
    np.testing.assert_allclose(combined['grad_m'], sum(row['grad_m'] for row in separate), rtol=1e-10, atol=1e-12)
    direction = np.random.default_rng(20261027).normal(size=small.n_nod)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    difference = (state(field*np.exp(step*direction), observations, 3*gamma)['objective']-
                  state(field*np.exp(-step*direction), observations, 3*gamma)['objective'])/(2*step)
    adjoint = float(combined['grad_m'] @ direction)
    np.testing.assert_allclose(adjoint, difference, rtol=1e-5, atol=1e-8)
    u, _ = solve_forward(small, cases[0].bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
    scaled, _ = solve_forward(small, 1.7*cases[0].bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
    np.testing.assert_allclose(scaled, 1.7*u, rtol=1e-9, atol=1e-11)
    return dict(seed=20261027, finite_difference_log_step=step, directional_adjoint=adjoint,
                directional_finite_difference=difference, sum_objective_and_gradient_checked=True,
                linear_force_scaling_checked=True)


def analyze(roots, output):
    if output.exists():
        raise FileExistsError('Choose a new diagnostic output; existing results are preserved')
    cfg, mesh, cases, audit, arrays, calibration = build_common_cases(roots)
    initial, initial_info = homogeneous_dic_initial(mesh, cases, cfg.THICKNESS, cfg.PRIMARY_NU)
    gamma_base = GAMMAS[-1]
    checks = kernel_checks(cfg, mesh, cases, gamma_base)
    output.mkdir(parents=True)
    (output/'implementation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(output/'observations.npz', **arrays, E_initial=initial, mesh_coordinates=mesh.coord)
    protocol = dict(scope='conditional_shared_modulus_feasibility; not formal gamma selection',
        source_cases=audit, geometry_mm=[mesh.geo_l, mesh.geo_h, cfg.THICKNESS], configured_elements=[121,19],
        physical_registration='Assumed common axial/transverse centers under earlier conditional placement; no matching landmarks.',
        independent_reference_images=True, shared_material_assumption='One time-invariant linear isotropic E(x,y), nu=0.26; different preload references assumed compatible.',
        force_reference='Unverified nominal end force minus record start force from each frozen input.',
        case_weights='Unmodified area weights; raw displacement units; no per-force or per-error normalization.',
        regularization='0.5*sum_c r_c.T W_c r_c + 0.5*gamma*integral(|grad E|^2)',
        diagnostic_gamma_rule='gamma = number of fit cases * base_gamma. This equals mean data cost + base_gamma/2*H1, preserves duplicate-case equivalence, and is not an L-curve choice.',
        base_gamma=gamma_base, initial=initial_info, max_iter=cfg.MAX_ITER, max_fun=cfg.MAX_FUN,
        ftol=cfg.FTOL, gtol=cfg.GTOL, nu=cfg.PRIMARY_NU, calibration_used_in_fit=False,
        implementation_sha256=file_sha256(Path(__file__)), kernel_checks=checks)
    (output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(audit).to_csv(output/'input_audit.csv', index=False, lineterminator='\n')
    # First joint solve is the smoke test; later fits only follow a successful replay.
    jobs = [('joint123_small', cases, 3*gamma_base),
            *[(f'single{index+1}', [case], gamma_base) for index, case in enumerate(cases)],
            ('joint23_small', cases[1:], 2*gamma_base),
            ('joint123_middle', cases, 3*GAMMAS[1]), ('joint123_large', cases, 3*GAMMAS[0])]
    measurements, solver_rows, profile_rows = [], [], []
    cal_values = pd.DataFrame(calibration).mean_E_MPa.to_numpy()
    for name, fit, gamma in jobs:
        destination = output/name
        destination.mkdir()
        print(f'{name}: fit={[case.name for case in fit]}, gamma={gamma:.8g}', flush=True)
        with (destination/'solver.log').open('x', encoding='utf-8') as stream, redirect_stdout(stream):
            result = _optimize(cfg, mesh, fit, gamma, cfg.PRIMARY_NU, initial.copy())
        field = result['E_final']
        np.savez_compressed(destination/'fields.npz', E_inverse=field, E_initial=initial,
                            cost_history=result['cost_history'], final_gradient=result['final_gradient'])
        residual_square = 0.
        for case in cases:
            u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
            state = evaluate_observation(case, u)
            if case.name in [item.name for item in fit]:
                residual_square += 2*state['cost_tar']
            measurements.append(dict(job=name, case=case.name, fitted=case.name in [item.name for item in fit], gamma=gamma,
                data_cost=state['cost_tar'], **relative_displacement_errors(case, state['prediction'])))
        np.testing.assert_allclose(np.sqrt(residual_square), result['residual_norm'], rtol=1e-8, atol=1e-10)
        profile = width_profile(mesh, field)
        solver_row = dict(job=name, gamma=gamma, fit_cases=[case.name for case in fit],
            converged=bool(result['converged']), stop_message=result['message'], iterations=result['n_iterations'],
            evaluations=result['n_evaluations'], final_gradient_inf=float(np.max(abs(result['final_gradient']))),
            gradient_tolerance_satisfied=bool(np.max(abs(result['final_gradient'])) <= cfg.GTOL),
            min_E_MPa=float(field.min()), max_E_MPa=float(field.max()),
            **calibration_metrics(mesh.plot_x[0]-mesh.geo_l/2, profile, cal_values))
        (destination/'metrics.json').write_text(json.dumps(solver_row, ensure_ascii=False, indent=2), encoding='utf-8')
        solver_rows.append(solver_row)
        profile_rows.extend(dict(job=name, x_mm=x, E_MPa=value) for x,value in zip(mesh.plot_x[0]-mesh.geo_l/2, profile))
        print(f"  stop={result['message']}; iterations={result['n_iterations']}", flush=True)
    measurements, solvers, profiles = pd.DataFrame(measurements), pd.DataFrame(solver_rows), pd.DataFrame(profile_rows)
    measurements.to_csv(output/'displacement_relative_errors.csv', index=False, lineterminator='\n')
    solvers.to_csv(output/'solver_and_calibration_metrics.csv', index=False, lineterminator='\n')
    profiles.to_csv(output/'profiles.csv', index=False, lineterminator='\n')
    render(output, mesh, measurements, solvers, profiles, cal_values)


def render(output, mesh, measurements, solvers, profiles, calibration):
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10, 'svg.fonttype':'path',
                         'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,1,figsize=(10,7), constrained_layout=True, sharex=True, sharey=True)
    styles = [('single1','第1次单独','#0072B2','--'), ('single2','第2次单独','#009E73','--'),
              ('single3','第3次单独','#CC79A7','--'), ('joint123_small','三次联合','#D55E00','-'),
              ('joint23_small','第2+3次联合','black',':')]
    for name,label,color,style in styles:
        data=profiles[profiles.job==name]
        axes[0].plot(data.x_mm,data.E_MPa,color=color,ls=style,lw=1.4,label=label)
    for name,label,color,style in [('joint123_large','γ=3×1.21e-8','#0072B2','-'),
                                  ('joint123_middle','γ=3×3.91e-9','#009E73','-'),
                                  ('joint123_small','γ=3×1.26e-9','#D55E00','-')]:
        data=profiles[profiles.job==name]
        successful = bool(solvers.loc[solvers.job==name,'converged'].iloc[0])
        axes[1].plot(data.x_mm,data.E_MPa,color=color,ls=style,lw=1.4,
                     label=label+('（异常停止，仅诊断）' if not successful else ''))
    for ax in axes:
        edges=np.r_[-mesh.geo_l/2,np.arange(-40.,41.,16.),mesh.geo_l/2]
        ax.plot(np.repeat(edges,2)[1:-1],np.repeat(calibration,2),color='.5',ls=':',lw=1,label='单调约束标定（仅外部比较）')
        ax.set_ylabel('宽度均值 E (MPa)')
        ax.legend(frameon=False,fontsize=8,ncol=3)
    axes[0].set_title('共同ROI：单独与联合识别；γ按拟合工况数同比增加')
    axes[1].set(title='三工况联合的固定γ敏感性（未执行L-curve选参）',xlabel='制造坐标 ξ (mm)')
    for suffix in ('.png','.svg'):
        fig.savefig(output/f'joint_profiles{suffix}',dpi=300)
    plt.close(fig)
    primary = measurements[measurements.job=='joint123_small'].set_index('case')
    paired = measurements[measurements.job=='joint23_small'].set_index('case')
    individual = measurements[measurements.job.isin(['single1','single2','single3']) & measurements.fitted].set_index('case')
    lines=['# 三次独立参考图的条件联合反演试算','',
        '**结论：** 可以在算法上共享一个自由二维模量场，但本次三个数据集与共同材料场假设的相容性不足。共同ROI单独识别时，第1/2/3次轴向误差为'+
        '/'.join(f'{individual.loc[name,"axial_relative_error_percent"]:.2f}%' for name in ('run1','run2','run3'))+
        '；三工况联合后变为'+ '/'.join(f'{primary.loc[name,"axial_relative_error_percent"]:.2f}%' for name in ('run1','run2','run3'))+
        '。不建议直接用三次联合结果替换各次结果。','',
        '第2+3次联合的轴向误差为'+ '/'.join(f'{paired.loc[name,"axial_relative_error_percent"]:.2f}%' for name in ('run2','run3'))+
        '，与两次单独解更相容。这个子集诊断是在看到工况差异后做的，不能作为预先独立验证或剔除第1次的依据。','',
        '共同二维模量场，三个独立位移解和刚体配准。所有结果限定于标线/横向中点对应、线弹性与共同材料性质假设；没有验证三个参考图中的材料点对应，也未同步核实起末载荷。不是正式四步联合识别或独立验证。','',
        f'共同域{mesh.geo_l:.9f}×{mesh.geo_h:.9f} mm，厚度3.2 mm，121×19自由二维H1，ν=0.26。取原三个中心对齐ROI的交集，保持毫米尺度，仅裁点与平移，不缩放；保留原位移和面积权重。每个工况端力按原牵引×共同宽度×厚度换算，保持其原牵引。','',
        '目标为0.5 Σ rᵀWr+0.5γ∫|∇E|²；各工况刚体模式分别消除。对照按γ=n_fit×γ_base配置，相当于平均数据项搭配同γ_base，但代码始终使用原始数据项之和；没有按力、噪声或相对误差重加权。本诊断采用预先给定三个γ_base，不宣称是联合L-curve最优。','',
        '三工况γ从3.79e-9提高到1.17e-8，三个轴向误差分别仅由14.425/4.911/3.475%变为14.433/4.933/3.481%，表明在已试范围内，相容性问题对γ不敏感。最高候选3.62e-8返回ABNORMAL，仅作诊断；其余6组因函数相对下降停止，7组均未满足GTOL=1e-12。','',
        '| 方案 | 评价工况 | 是否参与拟合 | 轴向相对误差 % | 横向相对误差 % | 双分量相对误差 % |',
        '|---|---|---|---:|---:|---:|']
    for r in measurements.itertuples():
        lines.append(f'| {r.job} | {r.case} | {r.fitted} | {r.axial_relative_error_percent:.3f} | {r.transverse_relative_error_percent:.3f} | {r.displacement_relative_error_percent:.3f} |')
    lines+=['','所有工况及标定参照此前均已用于诊断；未参与某个试算子集的工况只称跨工况检查，不能称预先留出的最终验证。','',
        '相同方向、相同边界形态的线弹性加载仅改变幅值，u_c∝牵引；它不会像改变加载方向那样提供新的力学激励。联合可以汇总重复观测、检验共同E假设，但不能自动消除参考载荷、厚度、定位、蠕变、损伤或DIC系统差异。','',
        '核验：共同ROI观测与原保留点逐元素一致；牵引换算守恒；8×3小网格三工况目标/梯度之和一致性、确定性log-E方向中心差分，以及1.7倍力的位移线性比例核验；每个识别场重新正演，联合残差范数核对到1e-8相对容差。停止标志与严格GTOL分别保存，不能以优化停止证明共同E假设成立。','',
        '![单独与联合模量曲线](joint_profiles.png)','',
        '复现：`python -u -B -m lib.joint_reference_diagnostic --cases <三个完整ROI冻结目录> --output <新输出目录>`。初值仅由三个工况DIC联合均匀拟合得到，标定不进入目标、初值或γ。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    assert file_sha256(output/'implementation_snapshot.py') == protocol['implementation_sha256']
    evidence=dict(optimization_implementation_sha256=protocol['implementation_sha256'],
        postprocessing_implementation_sha256=file_sha256(Path(__file__)),
        result_fields_sha256={str(path.relative_to(output)):file_sha256(path) for path in output.glob('*/fields.npz')},
        original_sources_verified_unchanged=True)
    for record in protocol['source_cases']:
        source=Path(record['source'])/'01_prepared'
        assert file_sha256(source/'fields.npz')==record['prepared_fields_sha256']
        assert file_sha256(source/'run_metadata.json')==record['prepared_metadata_sha256']
    (output/'postprocessing_provenance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(measurements.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,nargs=3)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report-only',action='store_true',help='Replot saved results; no optimization or raw-data changes')
    args=parser.parse_args()
    if args.report_only:
        protocol=json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
        length,width,_=protocol['geometry_mm']
        mesh=MeshInfo(length,width,121,19)
        source=Path(protocol['source_cases'][0]['source'])/'01_prepared/run_metadata.json'
        calibration=pd.DataFrame(json.loads(source.read_text(encoding='utf-8'))['calibration']).mean_E_MPa.to_numpy()
        render(args.output,mesh,pd.read_csv(args.output/'displacement_relative_errors.csv'),
               pd.read_csv(args.output/'solver_and_calibration_metrics.csv'),pd.read_csv(args.output/'profiles.csv'),calibration)
    else:
        if args.cases is None:
            parser.error('--cases is required for a new experiment')
        analyze(args.cases,args.output)
