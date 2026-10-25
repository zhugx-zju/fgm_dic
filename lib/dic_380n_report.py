"""Report saved 380N inversions with explicit method definitions and load caveats."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fgm_asm.mesh import MeshInfo
from lib.dic_exact_tv_report import _profile
from lib.experiment_data import file_sha256
from lib.dic_380n_spaces import profile,p
from lib.calibration_trend import add_calibration_trend

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/dic_full/load380_20261022'
METHODS=('h1','h1_uy10','huber1','huber10','exact_tv','exact_tv_fine')
LABELS={
 'h1':'节点连续模量＋梯度平方正则',
 'h1_uy10':'节点连续模量＋梯度平方；横向平方残差权重10',
 'huber1':'节点连续模量＋Huber-TV，δ=1 MPa/mm',
 'huber10':'节点连续模量＋Huber-TV，δ=10 MPa/mm',
 'exact_tv':'节点连续模量＋精确TV；121×19单元',
 'exact_tv_fine':'节点连续模量＋精确TV；242×38单元',
 'Q1_H1':'节点连续模量＋梯度平方；ψ=N',
 'Q1_TV':'节点连续模量＋Huber-TV(δ=10)；ψ=N',
 'P0_TV':'单元独立模量＋相邻单元差的Huber正则；ψ=N',
 'Q1_H1_PG':'节点连续模量＋梯度平方；ψ=N+0.25 h_x b ∂N/∂x',
 'P0_TV_PG':'单元独立模量＋相邻单元差的Huber正则；ψ=N+0.25 h_x b ∂N/∂x'}
BASELINES=dict(h1='coarse',h1_uy10='uy10',huber1='tv',huber10='tv/delta10',
 exact_tv='exact_tv_coarse_pd_20261021',exact_tv_fine='exact_tv_refine_242x38_20261021')

def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

def scalar(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)

def savefig(fig,stem):
    fig.savefig(stem.with_suffix('.png'),dpi=260)
    fig.savefig(stem.with_suffix('.svg'))
    plt.close(fig)

def collect(directory,method,fit_name):
    inverse=read(directory/'03_inverse/run_metadata.json')
    status=inverse.get('status','未开始')
    row=dict(method=method,label=LABELS[method],fit_platform=fit_name,status=status)
    if (directory/'03_inverse/selection.csv').exists():
        scan=pd.read_csv(directory/'03_inverse/selection.csv',float_precision='round_trip')
        row.update(attempted_gammas=len(scan),converged_gammas=int(scan.converged.sum()))
    final=directory/'03_inverse/nu_0.26/fields.npz'
    if not final.exists() or status not in ('computed','inverse_not_converged'):return row,None,None
    cfg=inverse['configuration'];mesh=MeshInfo(cfg['GEO_L'],cfg['GEO_H'],cfg['NEL_X'],cfg['NEL_Y'])
    with np.load(final) as saved:E=saved['E_inverse'].copy()
    mean,_=_profile(mesh,E);edges=mesh.plot_x[0]
    bounds=np.r_[0,cfg['REGION_BOUNDARIES_X'],cfg['GEO_L']]
    means=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        x=np.r_[a,edges[(edges>a)&(edges<b)],b]
        means.append(float(np.trapezoid(np.interp(x,edges,mean),x)/(b-a)))
    prepared=read(directory/'01_prepared/run_metadata.json')
    truth=np.array([r['mean_E_MPa'] for r in prepared['calibration']])
    solver=next(s for s in inverse['solvers'] if s['nu']==.26)
    row.update(gamma=inverse['selected_gamma'],mesh=[mesh.nel_x,mesh.nel_y],region_means_MPa=means,
        regional_mean_absolute_difference_percent=float(np.mean(abs(np.array(means)/truth-1))*100),
        gradient_inf=solver['final_gradient_inf'],converged=solver['converged'],
        gradient_tolerance_satisfied=solver.get('gradient_tolerance_satisfied'),stop_message=solver['message'])
    metrics=pd.read_csv(directory/'03_inverse/comparison.csv')
    row['metrics']={r['case']:dict(rmse_um=[r['rmse_ux_mm']*1000,r['rmse_uy_mm']*1000])
                    for r in metrics[np.isclose(metrics.nu,.26)].to_dict(orient='records')}
    common_path=directory/'verification/common_equal_weight_metrics.csv'
    if common_path.exists():
        common=pd.read_csv(common_path)
        if not common.empty:
            row['native_metrics']=row['metrics']
            row['metrics']={r['platform']:dict(rmse_um=[r['rmse_ux_mm']*1000,r['rmse_uy_mm']*1000])
                for r in common[np.isclose(common.nu,.26)].to_dict(orient='records')}
            row['metric_definition']='common equal-area weight and three-mode rigid fit'
    return row,mesh,E

def table(rows):
    return '<div class="scroll">'+pd.DataFrame(rows).to_html(index=False,escape=True,float_format=lambda x:f'{x:.6g}')+'</div>' if rows else '<p>尚无已保存结果。</p>'

def refresh_case_navigation():
    """Update only this experiment's navigation; preserve the project-level page."""
    titles={'01_prepared':'① 输入与核查','02_forward':'② 独立标定正演','03_inverse':'③ 反演及选参','04_comparison':'④ 三场及模量对照'}
    descriptions={'computed':'计算完成（数值条件见元数据）','prepared':'输入已冻结','running':'计算中',
        'l_curve_no_resolved_corner':'未形成有效拐点','l_curve_insufficient_corner_support':'拐点两侧有效支撑不足',
        'l_curve_insufficient_valid_points':'非零TV有效解不足','inverse_not_converged':'部分最终解未通过停止条件'}
    for method in METHODS:
        root=OUTPUT/method;prepared=read(root/'01_prepared/run_metadata.json')
        catalog=dict(case_id=prepared['case_id'],coordinates_verified=prepared['coordinates_verified'],stages={})
        page=['<html lang="zh-CN"><meta charset="utf-8"><title>'+LABELS[method]+'</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.7 Microsoft YaHei}li{margin:20px 0}</style>',
            '<h1>'+LABELS[method]+'</h1><p><a href="../index.html">返回380N各方案总对照</a></p>',
            '<p>380N假定319.9N；仅380N识别和选参。180N/280N为跨载荷诊断。物理配准和边界等效未独立验证。</p><ul>']
        for stage,title in titles.items():
            metadata=read(root/stage/'run_metadata.json')
            if not metadata:
                page.append('<li>'+title+'：未生成；上游选参未完成时不补指定最终模量。</li>');continue
            status=metadata['status'];catalog['stages'][stage]=dict(path=str(root/stage),status=status)
            links=[f'<a href="{stage}/run_metadata.json">完整配置与状态</a>']
            gallery='打开查看结果.html' if stage=='04_comparison' else '云图.html'
            if (root/stage/gallery).exists():links.append(f'<a href="{stage}/{gallery}">本步图表</a>')
            page.append('<li>'+title+'：'+descriptions.get(status,status)+'；'+' · '.join(links)+'</li>')
        (root/'结果目录.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
        (root/'结果总览.html').write_text(''.join(page)+'</ul></html>',encoding='utf-8')

def report(*, plot_only=False):
    if not plot_only:
        refresh_case_navigation()
    destination=OUTPUT/'comparison';destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
    prepared=read(OUTPUT/'h1/01_prepared/run_metadata.json')
    truth=np.array([r['mean_E_MPa'] for r in prepared['calibration']]);calibration_sd=np.array([r['sd_E_MPa'] for r in prepared['calibration']]);cfg=prepared['configuration']
    edges=np.r_[0,cfg['REGION_BOUNDARIES_X'],cfg['GEO_L']]
    summaries=[];states={};baselines={};pending=[]
    for method in METHODS:
        row,mesh,E=collect(OUTPUT/method,method,'380N');summaries.append(row)
        baselines[method]=collect(ROOT/'results/dic_full'/BASELINES[method],method,'180N')
        if E is not None:states[method]=(mesh,E)
        elif row['status'] in ('running','未开始'):pending.append(method)
    fig,axes=plt.subplots(3,2,figsize=(14,11),sharex=True,sharey=True,layout='constrained')
    for ax,method,row in zip(axes.flat,METHODS,summaries):
        ax.stairs(truth,edges,baseline=None,color='black',linestyle=':',label='独立标定均值（灰带±试样间SD，仅事后对照）')
        add_calibration_trend(ax, prepared['calibration'], cfg['REGION_BOUNDARIES_X'], cfg['GEO_L'], output=destination)
        ax.fill_between(np.repeat(edges,2)[1:-1],np.repeat(truth-calibration_sd,2),np.repeat(truth+calibration_sd,2),color='gray',alpha=.12)
        old,oldmesh,oldE=baselines[method]
        if oldE is not None:
            mean,sd=_profile(oldmesh,oldE)
            ax.plot(oldmesh.plot_x[0],mean,'--',color='#0072B2',label=f'180N识别，独立选γ={old["gamma"]:.3g}')
        if method in states:
            mesh,E=states[method];mean,sd=_profile(mesh,E)
            ax.plot(mesh.plot_x[0],mean,color='#D55E00',label=f'380N识别（假定319.9N），独立选γ={row["gamma"]:.3g}')
        else:ax.text(.5,.1,'380N未形成有效拐点，未指定γ' if row['status'].startswith('l_curve_') else '380N选参/求解尚未完成',transform=ax.transAxes,ha='center')
        ax.set(title=LABELS[method],xlabel='沿拉伸方向 x (mm)',ylabel='宽度平均杨氏模量 E (MPa)');ax.legend(fontsize=7.5)
    fig.suptitle('两种载荷分别选参：ν=0.26；380N暂取319.9N\n除细网格TV外为121×19单元；灰带为标定试样间标准差，不是识别置信区间',fontsize=12)
    savefig(fig,destination/'load_effect_profiles')
    if states:
        fig,axes=plt.subplots(len(states),1,figsize=(12,2.2*len(states)),sharex=True,layout='constrained',squeeze=False)
        lo=min(E.min() for mesh,E in states.values());hi=max(E.max() for mesh,E in states.values())
        for ax,(method,(mesh,E)) in zip(axes[:,0],states.items()):
            im=ax.pcolormesh(mesh.plot_x,mesh.plot_y,E.reshape(mesh.nods_y,mesh.nods_x),vmin=lo,vmax=hi,cmap='viridis',shading='gouraud')
            ax.set(title=LABELS[method],xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
        fig.colorbar(im,ax=list(axes[:,0]),label='杨氏模量 E (MPa)',shrink=.7)
        fig.suptitle('380N识别的自由二维模量：ν=0.26；载荷暂取319.9N；共用色标',fontsize=12)
        savefig(fig,destination/'modulus_fields')
    space_rows=[];space_states={}
    for method in p.METHODS:
        current=read(OUTPUT/'spaces'/method/'summary.json');control=read(OUTPUT/'spaces_180N_recovered'/method/'summary.json') or read(OUTPUT/'spaces_180N_control'/method/'summary.json')
        if not current:pending.append(method)
        if not control:pending.append(method+'_180N_control')
        space_rows.append(dict(method=method,label=LABELS[method],current=current,control=control))
        if current.get('status')=='selected':
            with np.load(OUTPUT/'spaces'/method/'selected.npz') as z:space_states[method]=z['E'].copy()
    if space_states:
        mesh=MeshInfo(cfg['GEO_L'],cfg['GEO_H'],42,8)
        fig,axes=plt.subplots(3,2,figsize=(14,11),sharex=True,sharey=True,layout='constrained')
        for ax,entry in zip(axes.flat,space_rows):
            method=entry['method'];ax.stairs(truth,edges,baseline=None,color='black',linestyle=':',label='独立标定均值（灰带±试样间SD，仅事后对照）')
            add_calibration_trend(ax, prepared['calibration'], cfg['REGION_BOUNDARIES_X'], cfg['GEO_L'])
            ax.fill_between(np.repeat(edges,2)[1:-1],np.repeat(truth-calibration_sd,2),np.repeat(truth+calibration_sd,2),color='gray',alpha=.12)
            for folder,row,color,line,load in [('spaces_180N_recovered' if (OUTPUT/'spaces_180N_recovered'/method/'selected.npz').exists() else 'spaces_180N_control',entry['control'],'#0072B2','--','180N'),('spaces',entry['current'],'#D55E00','-','380N，假定319.9N')]:
                if row.get('status')!='selected':continue
                with np.load(OUTPUT/folder/method/'selected.npz') as z:E=z['E']
                f=profile(mesh,E,method);label=f'{load}识别，50点重选γ={row["gamma"]:.3g}'
                if method.startswith('P0'):ax.stairs(f,np.linspace(0,cfg['GEO_L'],43),baseline=None,color=color,linestyle=line,label=label)
                else:ax.plot(mesh.plot_x[0],f,color=color,linestyle=line,label=label)
            if entry['control'] and entry['control'].get('status')!='selected':ax.text(.03,.08,'180N同50点控制：拐点支撑不足，未补指定γ',transform=ax.transAxes,fontsize=8)
            ax.set(title=LABELS[method],xlabel='沿拉伸方向 x (mm)',ylabel='宽度平均 E (MPa)');ax.legend(fontsize=7.3)
        axes.flat[-1].axis('off');axes.flat[-1].text(.03,.85,'共同条件：42×8位移单元；所有y行模量独立\n节点387个模量 / 单元336个模量\nν=0.26；两种载荷均独立扫描50个γ\nb=(1−s²)(1−t²)\n修改测试函数方案是bubble-PG，未采用SUPG',transform=axes.flat[-1].transAxes,va='top',fontsize=11)
        fig.suptitle('42×8单元：模量空间与测试函数对照；ν=0.26\n180N和380N均重扫50点γ；380N暂取319.9N',fontsize=12)
        savefig(fig,destination/'space_profiles')
    if plot_only:
        return
    material_rows=[];metric_rows=[];status_rows=[]
    for row in summaries:
        method=row['method'];old=baselines[method][0]
        status_rows.append({'方案':LABELS[method],'状态':row['status'],'已扫γ':row.get('attempted_gammas',0),'通过停止条件':row.get('converged_gammas',0)})
        if 'gamma' not in row:continue
        material_rows.append({'方案':LABELS[method],'380N所选γ':row['gamma'],'180N标定差异(%)':old.get('regional_mean_absolute_difference_percent'),'380N标定差异(%)':row['regional_mean_absolute_difference_percent'],'380N最终梯度∞范数':row['gradient_inf'],'达到原GTOL':row['gradient_tolerance_satisfied']})
        for name,result in row['metrics'].items():metric_rows.append({'方案':LABELS[method],'平台':name,'角色':'识别' if name=='380N' else '跨载荷诊断','轴向RMSE(μm)':result['rmse_um'][0],'横向RMSE(μm)':result['rmse_um'][1]})
    for entry in space_rows:
        row,old=entry['current'],entry['control']
        status_rows.append({'方案':LABELS[entry['method']]+'（42×8）','状态':row.get('status','计算中'),'已扫γ':row.get('count',0),'通过停止条件':row.get('successful_trials',0)})
        if row.get('status')=='selected':material_rows.append({'方案':LABELS[entry['method']]+'（42×8）','380N所选γ':row['gamma'],'180N标定差异(%)':old.get('regional_mean_absolute_difference_percent'),'380N标定差异(%)':row['regional_mean_absolute_difference_percent'],'380N最终梯度∞范数':row['selected_gradient_inf'],'达到原GTOL':row['selected_gtol']})
    pd.DataFrame(material_rows).to_csv(destination/'material_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(metric_rows).to_csv(destination/'displacement_comparison.csv',index=False,encoding='utf-8-sig')
    summary=dict(force_assumed_N=319.9,primary_nu=.26,production=summaries,baselines={k:v[0] for k,v in baselines.items()},spaces=space_rows,pending=pending,input_audit=read(OUTPUT/'input_audit/summary.json'),source_sha256=file_sha256(Path(__file__)))
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=scalar),encoding='utf-8')
    image_html=''.join(f'<figure><img src="comparison/{name}.png"><figcaption><a href="comparison/{name}.svg">矢量图</a></figcaption></figure>' for name in ('load_effect_profiles','modulus_fields','space_profiles','space_modulus_fields') if (destination/f'{name}.png').exists())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>380N：反演方案同条件对照</title><style>body{max-width:1320px;margin:30px auto;padding:0 24px;font:16px/1.7 "Microsoft YaHei",sans-serif}img{width:100%}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:7px}.scroll{overflow:auto}.warning{background:#fff4dc;padding:15px}a{color:#075fa2}</style><h1>380N：反演方案同条件对照</h1>
<p class="warning"><strong>探索载荷：319.9N = 示值380N − 偏置60.1N；缺少同步实测力。</strong>使用618—632共15帧；同初始参考图、23920个测点、原毫米坐标、真实截面20×3.2mm。载荷假设会影响模量绝对值。物理配准、截取域边界等效和跨批标定转移尚未完成独立验证。</p>
<p>各方案只用380N选择初值和γ，未把七区位置或独立标定写入反演先验。180N和280N为此前查看过的跨载荷诊断，不称为新的独立测试。主ν=0.26，其余0.24/0.28/0.33为固定γ敏感性。正式网格各50个γ；旧小网格方案另外补做180N同50点扫描。</p>
<h2>每种方案实际计算什么</h2><p>位移均采用四节点双线性插值。连续节点模量为 E=ΣNₐEₐ；单元独立模量为每个单元内 E=Eₑ，各单元可不同。原方案的数据项 f=½ΣAᵢ(rₓ²+rᵧ²)，横向十倍权重方案取f=½ΣAᵢ(rₓ²+10rᵧ²)。同一权重用于刚体配准、伴随与选参。</p>
<table><tr><th>方案</th><th>目标中的正则项</th><th>平衡方程测试函数ψ</th></tr>
<tr><td>节点连续模量＋梯度平方</td><td>(γ/2)∫[(∂ₓE)²+(∂ᵧE)²]dA</td><td>Nᵢ</td></tr>
<tr><td>节点连续模量＋Huber-TV</td><td>(γ/2)∫ρδ(√[(∂ₓE)²+(∂ᵧE)²])dA，δ=1或10 MPa/mm</td><td>Nᵢ</td></tr>
<tr><td>节点连续模量＋精确TV</td><td>γ∫√[(∂ₓE)²+(∂ᵧE)²]dA</td><td>Nᵢ</td></tr>
<tr><td>单元独立模量＋Huber正则</td><td>(γ/2)Σ相邻单元共享边 |F|ρδF(Eₑ−Eₑ′)，δF=10dF MPa</td><td>Nᵢ</td></tr>
<tr><td>修改测试函数的两个对照</td><td>分别沿用节点梯度平方或单元差的Huber正则</td><td>Nᵢ+0.25hₓ(1−s²)(1−t²)∂ₓNᵢ</td></tr></table>
<p>ρδ(z)=z²/(2δ)（|z|≤δ），否则ρδ(z)=|z|−δ/2。|F|是共享边长度，dF是相邻单元中心的法向间距。测试函数改动在正问题、转置伴随及材料导数中一致实现；属于已有bubble-PG对照，不是论文SUPG。</p>'''
    page+='<p>节点梯度平方方案已显示：380N场的七区均值更接近独立标定，但较低载荷的轴向预测误差增大。更高信噪比和模型/载荷/时间差异需要分别评价，不能只根据标定差异下降认定DIC精度是唯一原因。</p><h2>计算状态</h2>'+table(status_rows)+'<h2>模量与独立标定的差异</h2>'+table(material_rows)
    page+='<p>差异采用七区均值与独立标定均值之差的绝对值百分比再平均，区域积分按实际制造边界精确切分。该值不是对未知材料真值的误差。节点梯度/Huber的函数下降停止与GTOL满足分别报告；精确TV按驻点、对偶可行性、分裂残差和Fenchel间隙同时判定。</p>'
    page+=image_html+'<h2>380N识别场的跨载荷预测</h2><p>核验完成的方案统一用等面积权重和相同三个刚体模式评价；十倍横向权重仅属于其识别目标。各方案原始目标下的误差仍保存在各自四步结果中。</p>'+table(metric_rows)
    page+='<p>载荷比例位移差及逐帧变化见<a href="input_audit/summary.json">输入核查</a>；时间变化不全是纯DIC噪声，未除以√15。</p><h2>原四步结果与完整三场比较</h2><ul>'
    for method in METHODS:
        links=[f'<a href="{method}/结果总览.html">四步入口</a>']
        for relative,label in [('03_inverse/云图.html','选参和求解证据'),('04_comparison/打开查看结果.html','DIC／独立标定正演／反演回代')]:
            if (OUTPUT/method/relative).exists():links.append(f'<a href="{method}/{relative}">{label}</a>')
        page+=f'<li>{html.escape(LABELS[method])}：'+ ' · '.join(links)+'</li>'
    page+='</ul><h2>小网格五种空间方案：完整三场图</h2><ul>'
    for method in p.METHODS:
        if (OUTPUT/'spaces'/method/'云图.html').exists():page+=f'<li><a href="spaces/{method}/云图.html">{html.escape(LABELS[method])}</a></li>'
    page+='</ul><p><a href="space_verification.json">500场范数、梯度、选参和预测复核</a></p>'
    audit=read(OUTPUT/'input_audit/summary.json')
    quality=[]
    for row in audit.get('platforms',[]):
        quality.append({'平台':row['platform'],'实际/假设载荷(N)':row['force_N'],'轴向变形RMS(μm)':row['signal_rms_um'][0],
            '横向变形RMS(μm)':row['signal_rms_um'][1],'轴向去趋势帧间RMS(μm)':row['detrended_temporal_rms_um'][0],
            '横向去趋势帧间RMS(μm)':row['detrended_temporal_rms_um'][1]})
    scaling=[]
    with np.load(OUTPUT/'input_audit/fields.npz') as fields:
        target=fields['380N_mean_deformation_mm']
        for name in ('180N','280N'):
            source=fields[name+'_mean_deformation_mm']
            scale=float(np.sum(source*target)/np.sum(source*source))
            error=np.sqrt(np.mean((target-scale*source)**2,axis=0))*1000
            scaling.append({'对照':f'380N / {name}','两分量共同最小二乘幅值比':scale,
                '移除最佳幅值后轴向差异RMS(μm)':error[0],'移除最佳幅值后横向差异RMS(μm)':error[1]})
    (destination/'load_scaling.json').write_text(json.dumps(scaling,ensure_ascii=False,indent=2),encoding='utf-8')
    page+='<h2>位移信号及跨载荷比例检查</h2>'+table(quality)+table(scaling)
    page+='<p>以上幅值比仅描述观测场之间的比例，不用于重标定力、调整模量或选γ。去趋势帧间变化不是完整DIC不确定度，也没有按独立帧作√15缩减。单一幅值也不能消除的位移场差异，可能包含载荷、材料非线性、蠕变、相关误差及模型不一致。</p>'
    for method in ('exact_tv_fine',):
        record=read(OUTPUT/method/'03_inverse/run_metadata.json').get('numerical_continuation')
        if record:
            page+='<p class="warning">细网格TV数值协议：原均匀初值失败场及中断日志完整保留。后续用同380N已收敛粗场作数值初值；先对剩余γ作600次有界扫描，仅为五点曲率支撑补算至每点12,000次上限。只有通过原KKT条件的点参与最终选参；未通过的初筛点不能用于方法精度排名。不同网格总预算不相同。<a href="exact_tv_fine/03_inverse/multigrid/protocol.json">完整计算预算与初值来源</a></p>'
    page+='<p><a href="comparison/summary.json">完整指标和状态</a> · <a href="comparison/material_comparison.csv">模量比较CSV</a> · <a href="comparison/displacement_comparison.csv">位移比较CSV</a></p></html>'
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    print('Report:',OUTPUT/'index.html','pending:',pending,flush=True)

if __name__=='__main__':report()
