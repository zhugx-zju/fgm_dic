"""Reproduce the redistributable synthetic fixture; no experimental data are read."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgm_asm.mesh import MeshInfo
from fgm_asm.research import solve_forward, partition_modulus
from fgm_asm.observation import build_observation_operator


def generate(output: Path):
    if output.exists():
        raise FileExistsError(f'Fixture already exists: {output}')
    output.mkdir(parents=True)
    rng = np.random.default_rng(20261027)
    calibration = output/'calibration'
    calibration.mkdir()
    rows = []
    for infill, modulus in [(40, 900.), (70, 1400.), (100, 2100.)]:
        for repeat, factor in [(1, .99), (2, 1.01)]:
            strain = np.linspace(0, .01, 1201)
            force = 5+modulus*factor*strain*10*3.2
            name = f'{infill}_{repeat}.csv'
            pd.DataFrame(dict(time_s=np.arange(len(strain))*.02, force_N=force,
                displacement_mm=strain*50, extension_mm=strain*50)).to_csv(calibration/name, index=False)
            rows.append(dict(specimen_id=f'{infill}-{repeat}', infill_percent=infill,
                width_mm=10., thickness_mm=3.2, gauge_mm=50., file=name, block=1))
    pd.DataFrame(rows).to_csv(calibration/'specimens.csv', index=False)

    # camera_time = machine_time+2 s; two intermediate transitions at camera 7/12 s.
    def motion(t):
        return .01*t-.008*np.maximum(t-7, 0)+.008*np.maximum(t-12, 0)-.01*np.maximum(t-20, 0)

    timing = output/'timing'/'run1'
    timing.mkdir(parents=True)
    camera = np.arange(161)/5
    pd.DataFrame(dict(stage=np.arange(len(camera)), point1_DY_mm=motion(camera),
                     point2_DY_mm=motion(camera)+.02)).to_csv(timing/'disp.csv', index=False)
    machine = np.arange(1501)*.02
    displacement = motion(machine+2)
    force = 10+80*displacement/motion(np.array(20.))
    pd.DataFrame(dict(time_s=machine, force_N=force, displacement_mm=displacement,
                     extension_mm=displacement)).to_csv(timing/'force.csv', index=False)

    mesh = MeshInfo(48., 8., 12, 4)
    reference = partition_modulus([900., 1400., 2100.], [16., 32.])
    modulus = reference(mesh.X, mesh.Y)
    u, _ = solve_forward(mesh, 80., 3.2, .26, modulus)
    x, y = np.meshgrid(np.linspace(0, 48, 25), np.linspace(0, 8, 9))
    xy = np.column_stack((x.ravel(), y.ravel()))
    # Existing H is used unchanged; fixture noise produces a nonzero misfit floor.
    H = build_observation_operator(mesh, xy[:, 0], xy[:, 1])
    observed = np.asarray(H@u).reshape(-1, 2)
    systematic_noise = rng.normal(0, .00015, observed.shape)
    dic = output/'dic'/'run1'
    dic.mkdir(parents=True)
    for stage in range(129, 146):
        uv = observed+systematic_noise+rng.normal(0, .00002, observed.shape)
        frame = pd.DataFrame(dict(Type=np.ones(len(xy), dtype=int), X=xy[:, 0], Y=xy[:, 1],
            DX=uv[:, 0], DY=uv[:, 1], Disp_Combined=np.linalg.norm(uv, axis=1),
            EXX=np.zeros(len(xy)), EYY=np.zeros(len(xy)), EXY=np.zeros(len(xy)),
            grid_row=np.repeat(np.arange(9), 25), grid_col=np.tile(np.arange(25), 9)))
        frame.to_csv(dic/f'stage_{stage:05d}.comdat_comdata.csv', index=False)
    metadata = dict(kind='synthetic_demo_only', seed=20261027, units='mm, N, MPa',
        force_increment_N=80., poisson_ratio=.26, truth_representation='Q4 nodal layered samples',
        temporal_noise_sd_mm=.00002, common_spatial_noise_sd_mm=.00015,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='No real experimental input; calibration curves and timing are synthetic. '
             'The FE fixture demonstrates operation, not independent physical validation.')
    (output/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'data'/'example')
    generate(parser.parse_args().output)
