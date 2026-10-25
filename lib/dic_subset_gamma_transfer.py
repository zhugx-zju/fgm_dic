"""Controlled H1 gamma transfer on existing frozen local 380N observations.

Run each local case independently; this is a fixed-parameter diagnosis, not a
replacement L-curve or four-step experiment. Both gamma trials start from the
same locally fitted uniform E. Calibration is used only after optimization.
"""
import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fgm_asm.inverse_solver import _evaluate_inverse_state
from fgm_asm.load_case import evaluate_observation
from fgm_asm.material import MaterialInfo
from fgm_asm.research import partition_modulus, solve_forward
from .dic_case_workflow import load_prepared, homogeneous_dic_initial, _optimize, _solver_summary, _metrics
from .dic_preprocessing import save_dic_comparison
from .dic_subset_comparison import _common_regions, _rigid_metrics, _interpolator, NOTICE
from .experiment_data import file_sha256
from .research_reporting import save_run, plot_modulus_comparison, save_x_profile


ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT/'results/dic_full/load380_20261022/h1'
CASES = {'subset2': 'load380_subset2_60_50_20261025',
         'subset3': 'load380_subset3_60_50_40_20261025'}
DIAGNOSTIC = 'global_gamma_diagnostic_20261025'
REPORT = ROOT/'results/dic_full/subset380_gamma_transfer_20261025'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_case(case_name):
    root = ROOT/'results/dic_full'/CASES[case_name]
    source, output = root/'01_prepared', root/'03_inverse'/DIAGNOSTIC
    cfg, mesh, cases, metadata, provenance = load_prepared(source)
    previous = _read(root/'03_inverse/run_metadata.json')
    global_source = GLOBAL/'03_inverse/run_metadata.json'
    global_inverse = _read(global_source)
    if (cfg.REGULARIZATION_MODE != 'gradient' or
            getattr(cfg, 'MATERIAL_PARAMETERIZATION', 'nodal_2d') != 'nodal_2d' or
            global_inverse['configuration']['REGULARIZATION_MODE'] != 'gradient' or
            tuple(cfg.FIT_CASES) != ('380N',) or
            tuple(global_inverse['configuration']['FIT_CASES']) != ('380N',)):
        raise ValueError('Transfer requires the same platform and H1 definition')
    if global_inverse.get('selected_gamma') is None or global_inverse['l_curve_status'] != 'selected':
        raise ValueError('A genuinely selected global H1 gamma is required')
    fit = [case for case in cases if case.name == '380N']
    initial, initialization = homogeneous_dic_initial(mesh, fit, cfg.THICKNESS, .26)
    protocol = dict(workflow='fixed_gamma_transfer_diagnosis', status='controlled_trials', nu=.26,
        global_gamma=global_inverse['selected_gamma'], local_gamma=previous['selected_gamma'],
        global_gamma_source=str(global_source), global_gamma_source_sha256=file_sha256(global_source),
        gamma_selection='none: fixed global gamma plus same-initial local-gamma control',
        initialization=initialization, uses_global_modulus_as_initial=False, uses_calibration_to_select_gamma=False,
        number_of_trials=2, per_trial_max_iterations=cfg.MAX_ITER, per_trial_max_evaluations=cfg.MAX_FUN,
        boundary_and_observation_change=False, **provenance)
    save_run(output, cfg, protocol, {'initial_E': initial})
    audits = []
    reference = partition_modulus(pd.DataFrame(metadata['calibration']).mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    for tag, gamma in [('local_gamma_same_initial', previous['selected_gamma']),
                       ('global_gamma_same_initial', global_inverse['selected_gamma'])]:
        print(case_name, tag, 'gamma=', gamma, flush=True)
        result = _optimize(cfg, mesh, fit, gamma, .26, initial.copy())
        solver = _solver_summary(result)
        fields = dict(coordinates=mesh.coord, E_inverse=result['E_final'], initial_E=initial,
            final_gradient=result['final_gradient'], cost_history=result['cost_history'],
            cost_tar_history=result['cost_tar_history'], cost_reg_history=result['cost_reg_history'],
            gradient_norms=result['gradient_norms'])
        metrics = []
        for case in cases:
            u, _ = solve_forward(mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, .26, result['E_final'])
            state = evaluate_observation(case, u)
            fields[f'{case.name}_prediction'], fields[f'{case.name}_U'] = state['prediction'], u
            metrics.append(dict(platform=case.name, **_metrics(state)))
        run_cfg = SimpleNamespace(**vars(cfg))
        run_cfg.GAMMA, run_cfg.NU = gamma, .26
        destination = output/tag
        save_run(destination, run_cfg, dict(protocol, trial=tag, gamma=gamma, solver=solver,
            status='solver_stopped' if result['converged'] else 'not_converged'), fields)
        pd.DataFrame(metrics).to_csv(destination/'metrics.csv', index=False)
        fresh = _evaluate_inverse_state(mesh, fit[0].bc_info, None, MaterialInfo(nu=.26), gamma,
            result['E_final'], 0, observation_cases=fit)
        np.testing.assert_allclose(fresh['grad_m'], result['final_gradient'], rtol=1e-5, atol=2e-11)
        np.testing.assert_allclose(fresh['objective'], result['final_cost'], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(fresh['residual_norm'], result['residual_norm'], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(fresh['regularization_norm'], result['regularization_norm'], rtol=1e-10)
        audits.append(dict(trial=tag, gamma=gamma, independent_objective_and_gradient_reproduced=True,
            gradient_inf=float(np.max(abs(fresh['grad_m']))), gradient_tolerance_satisfied=solver['final_gradient_inf'] <= cfg.GTOL,
            converged=result['converged'], stop_message=result['message']))
        plot_modulus_comparison(destination, mesh, reference(mesh.X, mesh.Y), result['E_final'], identified_using='local 380N')
        save_x_profile(destination, mesh, result['E_final'], reference, identified_using='local 380N',
                       calibration=metadata['calibration'], region_boundaries=cfg.REGION_BOUNDARIES_X)
    (output/'verification.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    print('Saved:', output, flush=True)


def _uniform_information(prepared):
    """Gauss-Newton information in a uniform log(E) direction at E=1500 MPa.

    This is an analytical uniform-traction limiting case, not the full inverse
    Hessian or a measured uncertainty. Use actual point areas and rigid modes.
    """
    meta = _read(prepared/'run_metadata.json')
    cfg = meta['configuration']
    with np.load(prepared/'fields.npz') as data:
        xy, area = data['380N_coordinates'], data['380N_point_area_mm2']
        strain = meta['force_pairing']['380N']['mean_corrected_force_N']/(20*cfg['THICKNESS']*1500.)
        u = np.column_stack((strain*xy[:, 0], -.26*strain*xy[:, 1]))
        metric = _rigid_metrics(xy, np.zeros_like(u), u, area)
        return float(area.sum()*(metric['rmse_ux_um']**2+metric['rmse_uy_um']**2)/1e6)


def report():
    REPORT.mkdir(exist_ok=True)
    global_prepared = _read(GLOBAL/'01_prepared/run_metadata.json')
    global_information = _uniform_information(GLOBAL/'01_prepared')
    all_regions, all_metrics, statuses, checks, objectives = [], [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10., 3.8), sharey=True, layout='constrained')
    labels = {'previous_local_selected': 'Previous local L-curve result',
              'local_gamma_same_initial': 'Local gamma: common uniform start',
              'global_gamma_same_initial': 'Global gamma: common uniform start'}
    colors = {'previous_local_selected': '#999999', 'local_gamma_same_initial': '#0072B2',
              'global_gamma_same_initial': '#D55E00'}
    for (case_name, run), ax in zip(CASES.items(), axes):
        root = ROOT/'results/dic_full'/run
        cfg, mesh, cases, meta, _ = load_prepared(root/'01_prepared')
        output = root/'03_inverse'/DIAGNOSTIC
        protocol = _read(output/'run_metadata.json')
        offset = global_prepared['offset_mm'][0]-meta['offset_mm'][0]
        selected = _read(root/'03_inverse/run_metadata.json')
        directories = {'previous_local_selected': root/'03_inverse/nu_0.26',
                       'local_gamma_same_initial': output/'local_gamma_same_initial',
                       'global_gamma_same_initial': output/'global_gamma_same_initial'}
        with np.load(root/'01_prepared/fields.npz') as z:
            data = {k: z[k].copy() for k in z.files}
        with np.load(GLOBAL/'03_inverse/nu_0.26/fields.npz') as full:
            _, _, global_interpolator = _interpolator(full['coordinates'], full['E_inverse'])
            full_on_local = global_interpolator(np.column_stack((mesh.coord[:, 1], mesh.coord[:, 0]+offset)))
            fit = [case for case in cases if case.name == '380N']
            uniform_u, _ = solve_forward(mesh, fit[0].bc_info['tensile_end_force'], cfg.THICKNESS,
                                          .26, np.full(mesh.n_nod, 1500.))
            deformation = (evaluate_observation(fit[0], np.zeros(mesh.n_dof))['residual']-
                           evaluate_observation(fit[0], uniform_u)['residual'])
            fe_information = float(deformation @ (fit[0].observation.weight_vector*deformation))
            analytic_information = _uniform_information(root/'01_prepared')
            np.testing.assert_allclose(fe_information, analytic_information, rtol=1e-9, atol=1e-11)
            checks.append(dict(case=case_name, analytic_uniform_information_mm4=analytic_information,
                               FE_uniform_information_mm4=fe_information))
            objective_fields = {'full_field_sampled_on_local_Q4': full_on_local}
            for model, directory in directories.items():
                with np.load(directory/'fields.npz') as local:
                    objective_fields[model] = local['E_inverse'].copy()
                    table = _common_regions(full, local, offset, cfg.REGION_BOUNDARIES_X, meta['calibration'])
                    check = _common_regions(full, local, offset, cfg.REGION_BOUNDARIES_X, meta['calibration'], order=2)
                    difference = float(np.max(abs(table.difference_percent-check.difference_percent)))
                    assert difference < 1e-8
                    checks.append(dict(case=case_name, model=model, common_quadrature_difference_pp=difference,
                                       field_sha256=file_sha256(directory/'fields.npz')))
                    table = table[(table.model == 'subset_inverse') | (model == 'previous_local_selected')].copy()
                    table['model'] = table.model.replace({'subset_inverse': model})
                    table.insert(0, 'case', case_name)
                    all_regions.extend(table.to_dict(orient='records'))
                    for case in cases:
                        metric = _rigid_metrics(data[f'{case.name}_coordinates'], data[f'{case.name}_observed'],
                            local[f'{case.name}_prediction'].reshape(-1, 2), data[f'{case.name}_point_area_mm2'])
                        all_metrics.append(dict(case=case_name, model=model, platform=case.name, **metric))
                profile = pd.read_csv(directory/'x_profile.csv')
                ax.plot(profile.x_mm, profile.inverse_width_mean_E_MPa, color=colors[model],
                        linestyle=':' if model == 'previous_local_selected' else '-', label=labels[model])
                if model != 'previous_local_selected':
                    trial = _read(directory/'run_metadata.json')
                    solver = trial['solver']
                    statuses.append(dict(case=case_name, model=model, gamma=trial['gamma'],
                        converged=solver['converged'], stop_message=solver['message'],
                        gradient_tolerance_satisfied=solver['final_gradient_inf'] <= cfg.GTOL,
                        gradient_inf=solver['final_gradient_inf'], iterations=solver['n_iterations'],
                        evaluations=solver['n_evaluations']))
            for model, field in objective_fields.items():
                state = _evaluate_inverse_state(mesh, fit[0].bc_info, None, MaterialInfo(nu=.26),
                    protocol['global_gamma'], field, 0, observation_cases=fit)
                objectives.append(dict(case=case_name, model=model, gamma=protocol['global_gamma'],
                    data_cost=state['cost_tar'], regularization_cost=state['cost_reg'],
                    objective=state['objective'], gradient_inf=float(np.max(abs(state['grad_m'])))))
            np.savez_compressed(output/'sampled_full_field_for_objective_only.npz',
                                coordinates=mesh.coord, E_inverse=full_on_local)
        full_profile = pd.read_csv(GLOBAL/'03_inverse/nu_0.26/x_profile.csv')
        physical_x = np.unique(np.r_[offset, offset+mesh.geo_l,
            full_profile.x_mm[(full_profile.x_mm > offset) & (full_profile.x_mm < offset+mesh.geo_l)]])
        ax.plot(physical_x-offset, np.interp(physical_x, full_profile.x_mm, full_profile.inverse_width_mean_E_MPa),
                color='#7B3294', linestyle='-.', label='Full-domain result on same region')
        calibration = pd.DataFrame(meta['calibration'])
        cloud = directories['global_gamma_same_initial']/'displacement_comparison'
        if not (cloud/'380N/point_fields.csv').exists():
            with np.load(root/'02_forward/nu_0.26/fields.npz') as forward, \
                 np.load(directories['global_gamma_same_initial']/'fields.npz') as inverse:
                model_states = {}
                for model, saved in [('forward', forward), ('inverse', inverse)]:
                    prediction = saved['380N_prediction'].ravel()
                    residual = fit[0].observation.u_measured-prediction
                    weight = fit[0].observation.weight_vector
                    model_states[model] = [dict(prediction=prediction,
                        weighted_rms=float(np.sqrt(residual @ (weight*residual)/weight.sum())))]
                save_dic_comparison(cloud, fit, model_states, cfg.FIT_CASES,
                                    {'380N': data['380N_grid_ids']}, nu=.26)
        edges = np.repeat(np.r_[0., cfg.REGION_BOUNDARIES_X, mesh.geo_l], 2)[1:-1]
        ax.plot(edges, np.repeat(calibration.mean_E_MPa, 2), color='black', linestyle='--', label='Independent calibration')
        ax.fill_between(edges, np.repeat(calibration.mean_E_MPa-calibration.sd_E_MPa, 2),
                        np.repeat(calibration.mean_E_MPa+calibration.sd_E_MPa, 2), color='gray', alpha=.12)
        ax.set(xlabel='Local x (mm)', title='/'.join(map(str, cfg.REGION_INFILL_PERCENT)))
        checks.append(dict(case=case_name, uniform_logE_GN_information_mm4=_uniform_information(root/'01_prepared'),
            relative_to_global=_uniform_information(root/'01_prepared')/global_information,
            global_information_mm4=global_information,
            global_to_local_gamma_ratio=protocol['global_gamma']/protocol['local_gamma']))
    axes[0].set_ylabel('Width-mean E (MPa)')
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(REPORT/'gamma_transfer.png', dpi=300)
    fig.savefig(REPORT/'gamma_transfer.svg')
    plt.close(fig)
    regions, metrics, status = pd.DataFrame(all_regions), pd.DataFrame(all_metrics), pd.DataFrame(statuses)
    regions.to_csv(REPORT/'common_regional_modulus.csv', index=False)
    metrics.to_csv(REPORT/'common_displacement_metrics.csv', index=False)
    status.to_csv(REPORT/'solver_status.csv', index=False)
    objective_table = pd.DataFrame(objectives)
    objective_table.to_csv(REPORT/'local_objective_at_global_gamma.csv', index=False)
    summary = regions.assign(absolute_difference=regions.difference_percent.abs()).groupby(['case', 'model']).absolute_difference.mean().reset_index()
    summary.to_csv(REPORT/'material_summary.csv', index=False)
    (REPORT/'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    page = ['<!doctype html><meta charset="utf-8"><title>局部H1沿用全域γ诊断</title>',
        '<style>body{max-width:1300px;margin:30px auto;font:16px/1.7 sans-serif}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}</style>',
        '<h1>固定380N全域H1 γ：局部两区／三区诊断</h1>', '<p>'+html.escape(NOTICE)+'</p>',
        '<p>ν=0.26；全域γ=3.7275937203149377e−8。两个γ试验从同一本地DIC拟合的均匀初值出发，'
        '不使用全域E场作初值，不重扫γ、不改网格／观测／载荷／边界／停止设置。'
        '这是明确指定系数的诊断，不替换原局部L-curve结果；380N全域精确TV未选出有效γ，本次不迁移TV系数。</p>',
        '<p><a href="../../../docs/dic_subset_gamma_transfer_20261025.md">解释、协议、结果和核验</a></p>',
        summary.to_html(index=False, float_format=lambda v: f'{v:.6g}'),
        '<img src="gamma_transfer.png" alt="Same-initial local and global gamma comparison">',
        '<h2>求解停止证据</h2>', status.to_html(index=False, float_format=lambda v: f'{v:.6g}'),
        '<h2>共同积分区域差异</h2>', regions.to_html(index=False, float_format=lambda v: f'{v:.6g}'),
        '<h2>380N同点位移误差（μm）</h2>', metrics[metrics.platform == '380N'].to_html(index=False, float_format=lambda v: f'{v:.6g}'),
        '<h2>同一局部目标、全域γ下的代价对照</h2>',
        '<p>全域E只在求解后采样到局部Q4节点作代价评价，未用于局部初值。此项区分“局部目标偏好的场”与“标定更接近的场”，'
        '不是证明局部或全域物理边界正确。</p>',
        objective_table.to_html(index=False, float_format=lambda v: f'{v:.6g}'),
        '<p><a href="common_regional_modulus.csv">模量统计CSV</a> · <a href="common_displacement_metrics.csv">位移CSV</a> · '
        '<a href="verification.json">共同积分与均匀方向信息量检查</a></p>']
    for case_name, run in CASES.items():
        prefix = f'../{run}/03_inverse/{DIAGNOSTIC}/global_gamma_same_initial'
        page.append(f'<p>{case_name}：<a href="{prefix}/modulus_comparison.png">二维模量</a> · '
            f'<a href="{prefix}/fields.npz">固定全域γ的完整场</a> · '
            f'<a href="{prefix}/displacement_comparison/380N/380N_inverse_dic_components.png">DIC与固定γ回代</a> · '
            f'<a href="{prefix}/displacement_comparison/380N/380N_forward_dic_components.png">DIC与独立标定正演</a> · '
            f'<a href="{prefix}/displacement_comparison/380N/point_fields.csv">三场与残差逐点CSV</a></p>')
    (REPORT/'index.html').write_text('\n'.join(page), encoding='utf-8')
    print(summary.to_string(index=False)); print(status.to_string(index=False)); print(REPORT/'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=tuple(CASES))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if bool(args.case) == args.report_only:
        parser.error('Choose exactly one --case or --report-only')
    report() if args.report_only else run_case(args.case)
