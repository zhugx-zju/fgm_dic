"""Plot the saved 380N space probes without any new FE solves or inversion."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lib.dic_380n_report import OUTPUT,LABELS,savefig
from lib.dic_preprocessing import grid_fields_from_points
from lib.dic_380n_spaces import p


def plot():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
    saved={}
    for method in p.METHODS:
        path=OUTPUT/'spaces'/method/'selected.npz'
        if path.exists():
            with np.load(path) as z:saved[method]={k:z[k].copy() for k in z.files}
    if not saved:return
    low=min(z['E'].min() for z in saved.values());high=max(z['E'].max() for z in saved.values())
    fig,axes=plt.subplots(len(saved),1,figsize=(12,2.2*len(saved)),layout='constrained',squeeze=False,sharex=True)
    for ax,(method,z) in zip(axes[:,0],saved.items()):
        if method.startswith('P0'):
            values=z['E'].reshape(8,42);x=np.linspace(0,p.L,43);y=np.linspace(0,p.H,9)
            im=ax.pcolormesh(x,y,values,vmin=low,vmax=high,cmap='viridis',shading='flat')
        else:
            values=z['E'].reshape(9,43);x=np.linspace(0,p.L,43);y=np.linspace(0,p.H,9)
            im=ax.pcolormesh(x,y,values,vmin=low,vmax=high,cmap='viridis',shading='gouraud')
        ax.set(title=LABELS[method],xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    fig.colorbar(im,ax=list(axes[:,0]),label='杨氏模量 E (MPa)',shrink=.75)
    fig.suptitle('380N识别的二维模量：42×8位移单元；ν=0.26；载荷暂取319.9N；共用色标',fontsize=12)
    savefig(fig,OUTPUT/'comparison/space_modulus_fields')
    for method,z in saved.items():
        destination=OUTPUT/'spaces'/method
        links=[]
        for name in ('380N','180N','280N'):
            observed=z[name+'_observed'].reshape(-1,2);origin=observed.mean(0)
            fields=dict(dic=(observed-origin)*1000,
                forward=(z[name+'_forward_prediction'].reshape(-1,2)-origin)*1000,
                inverse=(z[name+'_prediction'].reshape(-1,2)-origin)*1000)
            raw={f'{model}_{component}':values[:,axis] for model,values in fields.items()
                 for axis,component in enumerate(('ux','uy'))}
            x,y,grids=grid_fields_from_points(z[name+'_coordinates'],raw)
            fig,axes=plt.subplots(3,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
            for col,component in enumerate(('ux','uy')):
                lower=min(np.min(grids[model+'_'+component]) for model in fields)
                upper=max(np.max(grids[model+'_'+component]) for model in fields)
                for row,(model,label) in enumerate((('dic','DIC观测'),('forward','独立标定正演'),('inverse','本方案反演回代'))):
                    ax=axes[row,col]
                    im=ax.pcolormesh(x,y,grids[model+'_'+component],vmin=lower,vmax=upper,cmap='jet',shading='nearest',rasterized=True)
                    ax.set(title=label+'：'+('轴向位移' if col==0 else '横向位移'),xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
                    ax.tick_params(labelbottom=True,labelleft=True)
                fig.colorbar(im,ax=list(axes[:,col]),label=component+' (μm)',shrink=.8)
            fig.suptitle(name+'；'+LABELS[method]+'；ν=0.26\n同一测点、参考图及显示平移；每个位移分量的三幅图共用色标',fontsize=12)
            stem=destination/(name+'_three_fields');savefig(fig,stem)
            links.append(f'<h2>{name}：'+('识别平台' if name=='380N' else '跨载荷诊断')+f'</h2><img src="{stem.name}.png"><p><a href="{stem.name}.svg">矢量图</a></p>')
        page='<html lang="zh-CN"><meta charset="utf-8"><title>'+LABELS[method]+'</title><style>body{max-width:1300px;margin:30px auto;font:16px/1.7 Microsoft YaHei}img{width:100%}</style><h1>'+LABELS[method]+'</h1><p>380N假定319.9N；仅380N选择初值和γ。180N/280N为此前查看过的跨载荷诊断。独立标定未参与反演。</p><p>原始位移未修改。图中统一减去该平台DIC均值，仅改变显示平移；FE仍使用原有两平移和一转动配准。显示单位为μm；模型各用其指定测试函数，不同方法的共同Galerkin误差另存于汇总JSON。</p>'+''.join(links)+'</html>'
        (destination/'云图.html').write_text(page,encoding='utf-8')
    print('Saved-only plots generated for',list(saved),flush=True)

if __name__=='__main__':plot()
