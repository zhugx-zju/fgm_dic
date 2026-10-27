"""Sequential DIC inversions: transfer only the previous modulus as initialization.

Each stage minimizes its current-case objective. Earlier residuals, gradients,
optimizer history, and prior-field penalties are not carried to the next stage.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import solve_forward
from .joint_reference_diagnostic import build_common_cases
from .dic_case_workflow import homogeneous_dic_initial, _optimize
from .dic_gamma_stability import GAMMAS, field_distance
from .dic_calibration_curve_comparison import width_profile, metrics as calibration_metrics
from .experiment_data import file_sha256
from .dic_preprocessing import relative_displacement_errors


def analyze(roots: list[Path], output: Path):
    if output.exists():
        raise FileExistsError('Use a new output directory; existing studies are preserved.')
    cfg, mesh, cases, audit, arrays, calibration = build_common_cases(roots)
    output.mkdir(parents=True)
    gamma = GAMMAS[-1]
    np.savez_compressed(output/'observations.npz', **arrays, mesh_coordinates=mesh.coord)
    calibration_values = pd.DataFrame(calibration).mean_E_MPa.to_numpy()
    uniform = {case.name: homogeneous_dic_initial(mesh, [case], cfg.THICKNESS, cfg.PRIMARY_NU) for case in cases}
    protocol = dict(source_cases=audit, gamma=gamma, common_geometry_mm=[mesh.geo_l, mesh.geo_h, cfg.THICKNESS],
        elements=[121,19], nu=cfg.PRIMARY_NU, max_iter_per_stage=cfg.MAX_ITER, max_fun_per_stage=cfg.MAX_FUN,
        ftol=cfg.FTOL, gtol=cfg.GTOL, stage_objective='0.5*r_current.T@W_current@r_current + 0.5*gamma*integral(|grad E|^2)',
        transfer='E_initial_next = E_final_previous, elementwise identical; fresh L-BFGS history each stage.',
        retained_earlier_objectives=False, penalty_to_previous_field=False,
        initial_information={name: info for name, (_, info) in uniform.items()},
        orders={'forward':['run1','run2','run3'],'reverse':['run3','run2','run1']},
        first_stage_reuse='Identical current-only initialization, target and budget: reuse its independent baseline result.',
        calibration_used_in_fit=False, selection='Fixed existing candidate for a protocol comparison; no formal gamma reselection.',
        coordinate_scope='Same conditional common-center registration as the preceding joint experiment; no physical point-registration certification.',
        budget_scope='Same budget per stage; a three-stage chain has three stage budgets, not one single-case total budget.',
        implementation_sha256=file_sha256(Path(__file__)),
        shared_preparation_implementation_sha256=file_sha256(Path(__file__).with_name('joint_reference_diagnostic.py')))
    (output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    solved, solver_records, evaluations = {}, [], []

    def fit(name, case, initial, initial_source):
        destination = output/name
        destination.mkdir()
        if initial_source != 'current_case_homogeneous_dic':
            with np.load(output/initial_source/'fields.npz') as prior:
                np.testing.assert_array_equal(initial, prior['E_inverse'])
        print(f'{name}: current={case.name}; initial={initial_source}; gamma={gamma:.8g}', flush=True)
        with (destination/'solver.log').open('x', encoding='utf-8') as stream, redirect_stdout(stream):
            result = _optimize(cfg, mesh, [case], gamma, cfg.PRIMARY_NU, initial.copy())
        field = result['E_final'].copy()
        np.savez_compressed(destination/'fields.npz', E_inverse=field, E_initial=initial,
            cost_history=result['cost_history'], final_gradient=result['final_gradient'])
        current_residual = None
        for observed_case in cases:
            displacement, _ = solve_forward(mesh, observed_case.bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
            state = evaluate_observation(observed_case, displacement)
            if observed_case.name == case.name:
                current_residual = np.sqrt(2*state['cost_tar'])
            evaluations.append(dict(job=name, optimized_case=case.name, evaluated_case=observed_case.name,
                fitted_current=observed_case.name == case.name,
                **relative_displacement_errors(observed_case, state['prediction'])))
        np.testing.assert_allclose(current_residual, result['residual_norm'], rtol=1e-8, atol=1e-10)
        independent = solved.get(f'independent_{case.name}')
        record = dict(job=name, optimized_case=case.name, initial_source=initial_source,
            gamma=gamma, converged=bool(result['converged']), message=result['message'],
            iterations=result['n_iterations'], evaluations=result['n_evaluations'],
            final_gradient_inf=float(np.max(abs(result['final_gradient']))),
            gradient_tolerance_satisfied=bool(np.max(abs(result['final_gradient'])) <= cfg.GTOL),
            field_difference_from_current_independent_percent=field_distance(mesh, field, independent['field']) if independent else 0.,
            **calibration_metrics(mesh.plot_x[0]-mesh.geo_l/2, width_profile(mesh, field), calibration_values))
        (destination/'metrics.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        solver_records.append(record)
        solved[name] = dict(field=field, record=record)
        print(f"  stop={result['message']}; iterations={result['n_iterations']}; vs independent={record['field_difference_from_current_independent_percent']:.6g}%", flush=True)
        return solved[name]

    # Three current-only baselines avoid using other cases even in initialization.
    for case in cases:
        fit(f'independent_{case.name}', case, uniform[case.name][0], 'current_case_homogeneous_dic')
    stages = []
    for label, order in [('forward', (0,1,2)), ('reverse', (2,1,0))]:
        first_name = f'independent_{cases[order[0]].name}'
        previous, previous_name = solved[first_name], first_name
        stages.append(dict(sequence=label, stage=1, job=first_name, optimized_case=cases[order[0]].name, reused_identical_first_stage=True))
        for index, case_index in enumerate(order[1:], start=2):
            if not previous['record']['converged']:
                stages.append(dict(sequence=label, stage=index, job='', optimized_case=cases[case_index].name,
                    status='not_executed: preceding solver did not report convergence'))
                break
            name = f'{label}_stage{index}_{cases[case_index].name}'
            previous = fit(name, cases[case_index], previous['field'], previous_name)
            stages.append(dict(sequence=label, stage=index, job=name, optimized_case=cases[case_index].name, reused_identical_first_stage=False))
            previous_name = name
    pd.DataFrame(stages).to_csv(output/'sequence_stages.csv', index=False, lineterminator='\n')
    pd.DataFrame(solver_records).to_csv(output/'solver_and_reference_metrics.csv', index=False, lineterminator='\n')
    pd.DataFrame(evaluations).to_csv(output/'cross_case_relative_errors.csv', index=False, lineterminator='\n')
    render(output, mesh, solved, stages, pd.DataFrame(evaluations))
    evidence = dict(implementation_sha256=protocol['implementation_sha256'],
        result_fields_sha256={str(p.relative_to(output)): file_sha256(p) for p in output.glob('*/fields.npz')},
        initial_transfer_checked_elementwise=True, current_target_residual_replayed=True,
        original_sources_verified_unchanged=True)
    for source in audit:
        root=Path(source['source'])/'01_prepared'
        assert file_sha256(root/'fields.npz') == source['prepared_fields_sha256']
        assert file_sha256(root/'run_metadata.json') == source['prepared_metadata_sha256']
    (output/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(solver_records)[['job','iterations','converged','field_difference_from_current_independent_percent']].to_string(index=False))


def render(output, mesh, solved, stages, errors):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'svg.fonttype':'path',
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,sharey=True,constrained_layout=True)
    for ax,label,title in zip(axes,('forward','reverse'),('顺序1→2→3','逆序3→2→1')):
        current=[stage for stage in stages if stage['sequence']==label and stage['job']]
        for stage,color in zip(current,('#0072B2','#009E73','#D55E00')):
            field=solved[stage['job']]['field']
            ax.plot(mesh.plot_x[0]-mesh.geo_l/2,width_profile(mesh,field),color=color,lw=1.5,
                    label=f"阶段{stage['stage']}，拟合{stage['optimized_case']}")
        last=current[-1]
        control=solved[f"independent_{last['optimized_case']}"]['field']
        ax.plot(mesh.plot_x[0]-mesh.geo_l/2,width_profile(mesh,control),color='black',ls='--',lw=1,
                label=f"{last['optimized_case']}独立初值对照")
        ax.set(title=title,ylabel='宽度均值 E (MPa)')
        ax.legend(frameon=False,fontsize=8,ncol=2)
    axes[-1].set_xlabel('制造坐标 ξ (mm)')
    for suffix in ('.png','.svg'):
        fig.savefig(output/f'sequential_profiles{suffix}',dpi=300)
    plt.close(fig)
    lines=['# 顺序工况反演：只传递上一模量场作为初值','',
        '按用户指定定义：E0→工况1识别E1→以E1初始化工况2识别E2→以E2初始化工况3识别E3；另做3→2→1逆序对照。每阶段只优化当前工况的数据项和同一个H1正则项，不保留先前残差、梯度或L-BFGS历史，也不添加向上一模量场靠拢的惩罚。','',
        '共同ROI、点集、权重、牵引、121×19自由二维网格和ν=0.26与前一次联合试算一致。γ固定1.2648552168552956e-9，不按工况数倍增，因为每阶段只有一个数据项。单独对照的均匀初值只由该工况DIC确定，顺序路径的首阶段复用完全相同的单独求解；后续模量初值逐元素核对与上一结果一致。共7次优化，形成3个单独对照与2条各3阶段路径。','',
        '| 路径 | 阶段 | 当前拟合工况 | 第1次轴向相对误差 % | 第2次轴向相对误差 % | 第3次轴向相对误差 % | 与当前工况单独解场差异 % |',
        '|---|---:|---|---:|---:|---:|---:|']
    for stage in stages:
        if not stage['job']:
            continue
        data=errors[errors.job==stage['job']].set_index('evaluated_case')
        distance=solved[stage['job']]['record']['field_difference_from_current_independent_percent']
        lines.append(f"| {stage['sequence']} | {stage['stage']} | {stage['optimized_case']} | "+
            ' | '.join(f'{data.loc[name,"axial_relative_error_percent"]:.4f}' for name in ('run1','run2','run3'))+
            f' | {distance:.6f} |')
    lines+=['','初值传递能影响优化路径、迭代数和可能的局部解，但当前目标不再包含先前工况的残差，因此不保证保留先前拟合。终点与最后一个工况单独解的差异是本实验检验的量，不预先假定必定相同。','',
        '每阶段预算1500迭代/12000目标评价，FTOL=1e-30、GTOL=1e-12；一条三阶段路径具有三个阶段预算，与一个单工况的总预算不同。不把函数停止当作严格一阶最优。若前阶段未报告求解成功，则后续阶段停止并保留失败结果。','',
        '三个独立参考图的共同材料坐标、首载与参考图对应及名义末载仍为原假设；这是顺序流程敏感性诊断，没有正式重新选γ，也不构成新的独立物理验证。标定只在求解结束后比较，逐工况交叉回代不称独立留出测试。','',
        '核验：初值传递逐元素一致；当前工况正演残差范数与优化器结果在1e-8相对容差内核对；原冻结输入哈希不变。','',
        '![顺序与逆序](sequential_profiles.png)','',
        '复现：`python -u -B -m lib.sequential_case_diagnostic --cases <三个完整ROI冻结目录> --output <新输出目录>`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.cases,args.output)
