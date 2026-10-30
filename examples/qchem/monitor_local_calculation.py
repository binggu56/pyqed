"""Monitor a local calculation, preserving sampled RSS, OS peak RSS and figures."""
import argparse
import csv
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import psutil


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (out/'memory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    minutes = [float(row['elapsed_s'])/60 for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(minutes, [float(row['rss_gib']) for row in rows], label='Calculation RSS')
    ax.plot(minutes, [float(row['available_gib']) for row in rows], label='System available RAM', alpha=.7)
    last = None
    for row, x in zip(rows, minutes):
        if row['stage'] != last:
            ax.axvline(x, color='grey', alpha=.3)
            ax.text(x, .98, row['stage'], rotation=90, va='top', fontsize=8,
                    transform=ax.get_xaxis_transform())
            last = row['stage']
    provenance = out/'provenance.json'
    metadata = json.loads(provenance.read_text()) if provenance.exists() else {}
    ax.set(xlabel='Elapsed time (minutes)', ylabel='GiB',
           title=metadata.get('figure_title', 'TDI cc-pVDZ SA6-CAS(6e,6o): local memory'))
    ax.legend(loc='lower right')
    fig.savefig(out/metadata.get('memory_figure', 'tdi_ccpvdz_cas66_memory.png'), dpi=150)
    plt.close(fig)
    history = out/'casscf_history.json'
    if history.exists():
        records = json.loads(history.read_text())
        if records:
            fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
            for name in ('energy', 'e_tot'):
                if name in records[0]:
                    ax.plot([r[name] for r in records], 'o-')
                    ax.set(xlabel='Macroiteration', ylabel='Energy (Hartree)', title='TDI CASSCF convergence')
                    fig.savefig(out/'tdi_ccpvdz_cas66_convergence.png', dpi=150)
                    break
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(out)
        return
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply a calculation command after --')
    if (out/'memory.csv').exists():
        raise FileExistsError('Use a fresh monitoring directory')
    started = time.monotonic()
    peak = 0.
    last_plot = started
    stage = 'STARTING'
    status = dict(stage=stage, rss_gib=0., available_gib=psutil.virtual_memory().available/2**30)
    with (out/'calculation.log').open('w', buffering=1) as log, (out/'memory.csv').open('w', buffering=1) as samples:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (out/'calculation.pid').write_text(str(child.pid)+'\n')
        process = psutil.Process(child.pid)
        writer = csv.writer(samples)
        writer.writerow(['elapsed_s', 'rss_gib', 'available_gib', 'stage'])
        while child.poll() is None:
            try:
                rss = process.memory_info().rss/2**30
            except psutil.NoSuchProcess:
                break
            available = psutil.virtual_memory().available/2**30
            peak = max(peak, rss)
            elapsed = time.monotonic()-started
            state = out/'stage.json'
            if state.exists():
                stage = json.loads(state.read_text())['stage']
            writer.writerow([elapsed, rss, available, stage])
            status = dict(pid=child.pid,
                          status='paused' if process.status() == psutil.STATUS_STOPPED else 'running', elapsed_s=elapsed,
                          stage=stage, rss_gib=rss, sampled_peak_rss_gib=peak, available_gib=available)
            temporary = out/'memory_status.tmp'
            temporary.write_text(json.dumps(status, indent=2)+'\n')
            temporary.replace(out/'memory_status.json')
            if time.monotonic()-last_plot >= 60:
                plot(out)
                last_plot = time.monotonic()
            time.sleep(1)
        code = child.wait()
    if (out/'stage.json').exists():
        status['stage'] = json.loads((out/'stage.json').read_text())['stage']
    status.update(status='complete' if code == 0 else 'failed', exit_code=code,
        elapsed_s=time.monotonic()-started, sampled_peak_rss_gib=peak,
        os_peak_rss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/
            (2**30 if sys.platform == 'darwin' else 2**20))
    (out/'memory_status.json').write_text(json.dumps(status, indent=2)+'\n')
    plot(out)
    print(json.dumps(status), flush=True)
    raise SystemExit(code if code >= 0 else 128-code)


if __name__ == '__main__':
    main()
