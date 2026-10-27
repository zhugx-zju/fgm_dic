"""Event-based synchronization, export candidates and explicit force pairing."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .dic_frame_selection import TimingData, read_timing, fit_transition, pair_force, draw_selection
from .experiment_data import file_sha256
from .results_io import save_run


def analyze_timing(data, spec, options):
    if not spec['zero_based_uninterrupted_confirmed']:
        raise ValueError('Confirm zero-based uninterrupted acquisition before mapping stages to time')
    offsets, events = [], []
    if spec.get('synchronization') == 'hardware':
        if not spec.get('synchronization_note', '').strip():
            raise ValueError('Hardware synchronization requires a provenance note')
        offsets = [float(spec['machine_time_offset_s'])]
    else:
        if len(spec['events']) < 2:
            raise ValueError('Use at least two independently identified synchronization events')
        for event in spec['events']:
            machine = fit_transition(data.machine[:, 0], data.machine[:, 2],
                                     event['machine_interval'], event['machine_search'], allow_jump=True)
            dic = [fit_transition(data.camera_time_s, y, event['dic_interval'],
                                  event['dic_search'], allow_jump=True)
                   for y in (*data.points[:, :, 1], data.mean_dy_mm)]
            differences = [machine['time_s']-fit['time_s'] for fit in dic]
            offsets.extend(differences)
            events.append(dict(machine=machine, dic_points_and_mean=dic, offsets_s=differences))
    offset = float(np.median(offsets))
    tail = fit_transition(data.camera_time_s, data.mean_dy_mm,
                          spec['tail_interval'], spec['tail_search'])
    if tail['slope_before_mm_s'] <= 0 or abs(tail['slope_after_mm_s']) > options['maximum_tail_slope_ratio']*tail['slope_before_mm_s']:
        raise ValueError('Configured final transition does not identify a positive-loading hold')
    entries = [fit_transition(data.camera_time_s, y,
        tuple(np.asarray(spec['tail_interval'])+shift), spec['tail_search'])['time_s']
        for shift in options['sensitivity_shifts_s'] for y in (*data.points[:, :, 1], data.mean_dy_mm)]
    # Ignore sub-nanoframe roundoff when a fitted event lies exactly on a frame.
    first = int(np.ceil((tail['time_s']+options['delay_after_entry_s'])*data.fps-1e-9))
    count = options['frame_count']
    if not isinstance(count, int) or count < 1:
        raise ValueError('FRAME_SELECTION_OPTIONS.frame_count must be a positive integer')
    last = first+count-1
    if first < 0 or last >= len(data.camera_time_s):
        raise ValueError('DIC does not cover the requested maximum-load hold window')
    if (first/data.fps < max(entries)+options['minimum_settle_s'] or
            last/data.fps > tail['time_s']+options['latest_after_entry_s']):
        raise ValueError('Window violates settling or latest-time limits; inspect transition sensitivity')
    stages = np.arange(first, last+1)
    machine_time = data.camera_time_s+offset
    paired, covered = pair_force(data, machine_time)
    robust_coverage = all(pair_force(data, data.camera_time_s[stages]+value)[1].all()
                          for value in (min(offsets), max(offsets)))
    complete = bool(covered[stages].all() and robust_coverage)
    # A sparse machine record must not silently interpolate over a long gap.
    gaps = np.diff(data.machine[:, 0])
    lower = first/data.fps+min(offsets)
    upper = last/data.fps+max(offsets)
    overlaps = (data.machine[:-1, 0] <= upper) & (data.machine[1:, 0] >= lower)
    gap_ok = not np.any(gaps[overlaps] > options['maximum_force_gap_s'])
    force_range = float(np.ptp(paired[stages])) if complete else None
    mean = float(paired[stages].mean()) if complete else None
    stable = bool(complete and gap_ok and force_range/max(abs(mean), 1e-15) <= options['maximum_force_range_fraction'])
    peak_index = int(np.argmax(data.machine[:, 1]))
    maximum_hold = bool(complete and abs(mean-data.machine[peak_index, 1]) <=
                        options['maximum_peak_difference_fraction']*max(abs(data.machine[peak_index, 1]), 1e-15))
    reference = spec.get('reference_force_N')
    reference_ok = reference is not None and np.isfinite(reference) and bool(spec.get('reference_note', '').strip())
    corrected = mean-float(reference) if stable and maximum_hold and reference_ok else None
    ready = corrected is not None and corrected > 0
    return dict(run=spec['run'], setting_N=spec['setting_N'], machine_time_offset_s=offset,
        event_offset_range_s=[float(min(offsets)), float(max(offsets))], events=events, tail_fit=tail,
        tail_entry_sensitivity_s=[float(min(entries)), float(max(entries))],
        stage_first=first, stage_last=last, frame_count=count, midpoint_stage=(first+last)//2,
        raw_peak_N=float(data.machine[peak_index, 1]), peak_machine_time_s=float(data.machine[peak_index, 0]),
        peak_approximate_stage=float((data.machine[peak_index, 0]-offset)*data.fps),
        force_coverage_count=int(covered[stages].sum()), full_force_coverage=complete,
        force_gap_check=gap_ok, force_stable=stable, near_maximum_load=maximum_hold,
        force_mean_N=mean, force_range_N=force_range,
        reference_force_N=float(reference) if reference_ok else None, reference_note=spec.get('reference_note'),
        mean_corrected_force_N=float(corrected) if ready else None,
        status='ready' if ready else 'export_candidate_only',
        selection_rule=f'entry+{options["delay_after_entry_s"]:g} s; {count} frames; {data.fps:g} Hz'), paired, covered


def select_frames(cfg):
    output = Path(cfg.FRAME_SELECTION_OUTPUT)
    if output.exists():
        raise FileExistsError(f'Choose a new RUN_NAME: {output}')
    records, requests, sources, data_records = [], [], {}, []
    for spec in cfg.TIMING_RUNS:
        folder = Path(spec['folder'])
        force_file, disp_file = spec['force_file'], spec['displacement_file']
        data = read_timing(folder, spec['fps'], force_file=force_file, displacement_file=disp_file)
        record, paired, covered = analyze_timing(data, spec, cfg.FRAME_SELECTION_OPTIONS)
        records.append(record); data_records.append((data, record))
        for filename in (force_file, disp_file):
            path = folder/filename
            sources[str(path)] = file_sha256(path)
        for stage in range(record['stage_first'], record['stage_last']+1):
            requests.append(dict(run=spec['run'], stage=stage, expected_filename=f'stage_{stage:05d}.comdat_comdata.csv',
                camera_time_s=stage/data.fps, estimated_machine_time_s=stage/data.fps+record['machine_time_offset_s'],
                paired_raw_force_N=float(paired[stage]) if covered[stage] else None,
                status=record['status']))
    arrays = {}
    for data, record in data_records:
        key = f'run{record["run"]}'
        arrays.update({key+'_machine': data.machine, key+'_points': data.points,
                       key+'_camera_time_s': data.camera_time_s, key+'_fps': np.array(data.fps)})
    save_run(output, cfg, dict(workflow='dic_frame_selection', status='computed', runs=records,
        source_hashes=sources, limitation='Event spread is sensitivity, not hardware synchronization accuracy. '
        'Two-point motion does not establish full-field stability; force and image references require separate evidence.'), arrays)
    pd.DataFrame(records).drop(columns=['events', 'tail_fit']).to_csv(output/'selected_dic_windows.csv', index=False)
    pd.DataFrame(requests).to_csv(output/'export_requests.csv', index=False)
    for data, record in data_records:
        times = data.camera_time_s+record['machine_time_offset_s']
        paired, covered = pair_force(data, times)
        pd.DataFrame(dict(stage=np.arange(len(times)), camera_time_s=data.camera_time_s,
            estimated_machine_time_s=times, mean_DY_mm=data.mean_dy_mm, raw_force_N=paired,
            force_record_coverage=covered)).to_csv(output/f'run{record["run"]}_timing.csv', index=False)
    plot_frame_selection_results(output)
    return records


def plot_frame_selection_results(output):
    """Read frozen timing arrays and selection; never rerun event detection."""
    output = Path(output)
    metadata = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    cfg = metadata['configuration']
    with np.load(output/'fields.npz') as saved:
        for record in metadata['runs']:
            key = f'run{record["run"]}'
            data = TimingData(saved[key+'_machine'], saved[key+'_points'], ['point1-DY','point2-DY'],
                              saved[key+'_camera_time_s'], float(saved[key+'_fps']))
            for language in cfg['FIGURE_LANGUAGES']:
                draw_selection(data, record, output, language, cfg['SPECIMEN_LABEL'], cfg['FORCE_PLOT_LIMITS'])
