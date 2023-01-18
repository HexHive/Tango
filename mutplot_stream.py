import argparse
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import collections.abc
from itertools import chain
from dataclasses import dataclass
from struct import unpack_from, calcsize
from mmap import mmap, PROT_READ

@dataclass(kw_only=True)
class Context:
    start_ts: float = None
    fig: plt.Figure = None
    ax: plt.Axes = None
    ax2: plt.Axes = None

    end_ts = None
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sorted_action_labels = None
    history = None
    ax_legend = None
    ax2_legend = None
    ax2_max_features = 0. 
    ax2_max_cum_features = 0.

@dataclass
class Log:
    mm: mmap
    off: int

    def __post_init__(self):
        self._labels = []
        self.unpack_header()

    def unpack(self, fmt):
        sz = calcsize(fmt)
        res = unpack_from(fmt, self.mm, self.off)
        self.off += sz
        return res

    def unpack_header(self):
        lc, = self.unpack('B')
        for _ in range(lc):
            llen, = self.unpack('B')
            lbl, = self.unpack(f'{llen}s')
            self._labels.append(lbl.decode())

    def unpack_entry(self):
        ts, mcount = self.unpack('dI')
        models = dict()
        for _ in range(mcount):
            lbl, model = self.unpack_model()
            models[lbl] = model
        return ts, models

    def unpack_model(self):
        llen, = self.unpack('B')
        lbl, = self.unpack(f'{llen}s')
        model = {
            'features': self.unpack('I')[0],
            'cum_features': self.unpack('I')[0],
            'actions': {
                x: self.unpack('ff') for x in self._labels
            }
        }
        return lbl.decode(), model

def period(astr):
    if astr.lower().endswith('m'):
        units = 'timedelta64[m]'
    elif astr.lower().endswith('h'):
        units = 'timedelta64[h]'
    elif astr.lower().endswith('d'):
        units = 'timedelta64[D]'
    else:
        raise argparse.ArgumentTypeError('wrong time units')
    return np.array(astr[:-1], dtype=units).tolist().total_seconds()

def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Plots the weight distribution timeseries for mutator schedules."
    ))
    parser.add_argument("logfile",
        help="The path to the model_history.json log file.")
    parser.add_argument("outdir",
        help="The path to the directory where figures will be saved.")
    parser.add_argument("--duration", "-D", default='1d', type=period,
        help="The duration of the fuzzing campaign to be plotted.")
    parser.add_argument("--batch", "-B", default=64, type=int,
        help="The number of datapoints to batch together when plotting.")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help=("Controls the verbosity of messages. "
            "-v prints info. -vv prints debug. Default: warnings and higher.")
        )

    return parser.parse_args()

def configure_verbosity(level):
    mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    # will raise exception when level is invalid
    numeric_level = mapping[level]
    logging.getLogger().setLevel(numeric_level)

def pp_time(time):
    if np.isnan(time) :
        return time
    if time < 60:
        return '%.fs' % time
    if time < (60 * 60):
        return '%.1fm' % (time / 60)
    if time < (24 * 60 * 60):
        return '%.1fh' % (time / (60 * 60))
    if time < (7 * 24 * 60 * 60):
        return '%.1fd' % (time / (24 * 60 * 60))
    if time < (30 * 24 * 60 * 60):
        return '%.1fw' % (time / (7 * 24 * 60 * 60))
    return '%.1fM' % (time / (30 * 24 * 60 * 60))

def list_ticks(bound, *, logfactor=None, linterval=None):
    if logfactor is not None:
        DENOMINATIONS = [
            1 * 60, # minutes
            15 * 60, # quarter-hour
            30 * 60, # half-hour
            60 * 60, # hour
            12 * 60 * 60, # half-day
            24 * 60 * 60, # day
            7 * 24 * 60 * 60, # week
            30 * 24 * 60 * 60, # month
        ]
        current_denom = 0
        last_tick = min(DENOMINATIONS[current_denom], bound)
        ticks = [last_tick]
        while last_tick < bound:
            last_tick *= logfactor
            if (current_denom + 1) < len(DENOMINATIONS) \
                and last_tick >= DENOMINATIONS[current_denom + 1]:
                current_denom += 1
                last_tick = DENOMINATIONS[current_denom]
            ticks.append(last_tick)
    elif linterval is not None:
        if linterval <= 0:
            linterval = bound // 20
        last_tick = min(linterval, bound)
        ticks = [last_tick]
        while last_tick < bound:
            last_tick += linterval
            ticks.append(last_tick)
    else:
        raise ValueError("One of logfactor or linterval must be specified")
    return ticks

def plot_state_chunk(ctx, ts_dict):
    # result: list(tuple(timestamp, weights_dict))
    sorted_points = sorted(ts_dict.items(), key=lambda x: float(x[0]))
    timestamps, models = zip(*sorted_points)
    timestamps = [float(x) - ctx.start_ts for x in timestamps]

    if (sorted_labels := getattr(ctx, 'sorted_action_labels', None)) is None:
        sorted_labels = ctx.sorted_action_labels = sorted(models[0]['actions'].keys())

    weights = zip(*([x['actions'][y][1] for y in sorted_labels] for x in models))
    features = list(map(lambda x: x['features'], models))
    cum_features = list(map(lambda x: x['cum_features'], models))

    ctx.ax2_max_features = max(ctx.ax2_max_features, *features)
    ctx.ax2_max_cum_features = max(ctx.ax2_max_cum_features, *cum_features)
    ctx.end_ts = timestamps[-1]

    if ctx.history is not None:
        timestamps.insert(0, ctx.history['timestamp'])
        hist_weights = [ctx.history['actions'][y][1] for y in sorted_labels]
        weights = zip(hist_weights, *zip(*weights))
        features.insert(0, ctx.history['features'])
        cum_features.insert(0, ctx.history['cum_features'])

    ctx.history = {
        'timestamp': timestamps[-1],
        'actions': models[-1]['actions'],
        'features': models[-1]['features'],
        'cum_features': models[-1]['cum_features']
    }

    # plot probabilities
    polys = ctx.ax.stackplot(timestamps, *weights, labels=sorted_labels, colors=ctx.colors)
    if ctx.ax_legend is None:
        ctx.ax_legend = ctx.ax.get_legend_handles_labels()
    for poly in polys:
        ctx.ax.draw_artist(poly)
        poly.remove()

    # plot feature counts
    # Unlike the stackplot, feature counts are drawn later since there is no
    # known or tangible upper bound; this results in a higher memory usage but
    # saves from a lot of headache.
    lines = ctx.ax2.plot(timestamps, features, color='black', linestyle='solid', drawstyle='steps-post', label="Local features")
    lines = ctx.ax2.plot(timestamps, cum_features, color='black', linestyle='dashed', drawstyle='steps-post', label="Accumulated features")
    if ctx.ax2_legend is None:
        ctx.ax2_legend = ctx.ax2.get_legend_handles_labels()

def plot_batch(plots, batch, start_ts=None, duration=24*60*60, logscale=False):
    if start_ts is None:
        start_ts = min(min(float(x) for x in ts_dict.keys()) for ts_dict in batch.values())

    duration_exceeded = True
    for state, ts_dict in batch.items():
        filtered_dict = {
            ts: models for ts, models in ts_dict.items()
            if float(ts) - start_ts <= duration
        }
        if filtered_dict:
            duration_exceeded = False
        else:
            continue
        if (ctx := plots.get(state)) is None:
            fig = plt.figure(figsize=(20,10))
            ax = plt.axes([.035, .05, .74, .93])
            if logscale:
                xticks = list_ticks(duration, logfactor=2)
                ax.set_xscale('symlog')
            else:
                xticks = list_ticks(duration, linterval=0)
            xticklabels = [pp_time(x) for x in xticks]

            ax.set_xticks(xticks, minor=False)
            ax.set_xticklabels(xticklabels)
            ax.set_xlim(left=0, right=duration)
            ax.set_xlabel("Time")

            ax.set_ylim(bottom=0, top=1)
            ax.set_ylabel("Probability")

            # we draw the canvas without the second axes
            fig.canvas.draw()

            ax2 = ax.twinx()
            ax2.set_ylabel("Features")
            ax2.set_ylim(bottom=0)

            ctx = plots[state] = Context(
                start_ts=start_ts,
                fig=fig,
                ax=ax,
                ax2=ax2)

        plot_state_chunk(ctx, filtered_dict)

    if duration_exceeded:
        # all timestamps are beyond the requested duration
        return None
    else:
        return start_ts

def transpose(nested_dict):
    result = dict()
    for x, nested in nested_dict.items():
        for a, val in nested.items():
            result.setdefault(a, dict())[x] = val
    return result

def main():
    args = parse_args()
    configure_verbosity(args.verbose)

    plots = {}
    start_ts = None
    count = 0
    with open(args.logfile, "rb") as f:
        mm = mmap(f.fileno(), 0, prot=PROT_READ)

    log = Log(mm, 0)
    batch = []
    # parse entries
    while log.off < mm.size():
        try:
            batch.append(log.unpack_entry())
        except Exception:
            break
        if len(batch) >= args.batch:
            dbatch = transpose(dict(batch)) # temporary, should keep timestamps as first index
            batch.clear()
            if (ret := plot_batch(plots, dbatch, start_ts, logscale=False, duration=args.duration)):
                start_ts = ret
            else:
                break
        count += 1
        if count % args.batch == 0:
            print(f'Processed {count} chunks')
    if batch:
        dbatch = transpose(dict(batch))
        plot_batch(plots, dbatch, start_ts, logscale=False, duration=args.duration)

    for label, ctx in plots.items():
        print(f'Saving {label}.png')
        renderer = ctx.fig.canvas.renderer

        # we draw ax2 because it was never explictly drawn
        ctx.ax2.autoscale(axis='y')
        ctx.ax2.set_ylim(bottom=0)

        ctx.ax2.hlines(y=ctx.ax2_max_features, xmin=ctx.end_ts, xmax=args.duration, linestyles='--', lw=2, colors='lightgrey')
        ctx.ax2.hlines(y=ctx.ax2_max_cum_features, xmin=ctx.end_ts, xmax=args.duration, linestyles='--', lw=2, colors='lightgrey')

        ctx.ax2.set_yticks(list(ctx.ax2.get_yticks()) + [ctx.ax2_max_features, ctx.ax2_max_cum_features])

        ctx.ax2.draw(renderer)

        # then we draw the legends
        handles, labels = ctx.ax_legend
        leg = ctx.ax.legend(handles[::-1], labels[::-1], title="Mutators", bbox_to_anchor=(1.05, 1), loc="upper left")
        leg.draw(renderer)

        handles, labels = ctx.ax2_legend
        leg2 = ctx.ax2.legend(handles, labels, title="Features", bbox_to_anchor=(1.05, 0), loc="lower left")
        leg2.draw(renderer)

        PIL.Image.frombytes('RGBA',
            (int(renderer.width), int(renderer.height)),
            renderer.buffer_rgba().tobytes()).save(
                os.path.join(args.outdir, f'{label}.png'), format='png')

        plt.close(ctx.fig)

    # plot(history, args.outdir, logscale=False)

if __name__ == '__main__':
    main()
