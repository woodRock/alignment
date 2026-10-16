"""Step 13 — one summary over calibration seeds for the paper's tables.

Reads, for every pair, the arrays of steps 05 (icoshift, COW-TIC, resampling control), 07
(constant shift, linear warp, PCHIP), 06 (WarpTransformer, per seed), 08 (pretraining
ablation, per seed) and 11 (within-batch ceiling), and reports per pair:
  - consistency checks: the deterministic arrays every seed's file carries (unaligned, the
    anchor baselines, anchors) are identical across seeds and match steps 05 and 07
  - every TIC metric of every method (evaluation.METRICS: normalised, raw, both smoothed with
    σ = 1 and 2 bins, r in range, non-empty bins); the WarpTransformer as mean ± s.d.
  - coverage (outside), study silhouette and warp magnitude of every method
  - the ranking of the methods on each metric
  - WT − each baseline on each metric over seeds, and the number of seeds at which it is > 0
  - each method's share of the gap to the within-batch ceiling (normalised and raw σ1)
  - per batch-A sample (r averaged over B_j): how many the WT beats COW-TIC for, per seed
  - warp magnitudes before and after calibration
  - the ablation: both arms' TIC r and warps (before and after), the pretraining gain per seed,
    and the share of the improvement over unaligned that random init recovers
The four evaluated pairs come first; the two excluded (control) pairs follow, seed 0 only.
s.d. is the sample s.d. over seeds (ddof = 1).

Output: results/seed_summary.txt (exit status 1 if a check fails)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._compat import mean_sd, per_a_mean
from ..config import EVAL_PAIRS, EXCLUDED_PAIRS, Paths, pair_label
from ..evaluation import METRICS
from ._common import parser, setup_logging, table, title, write_report

Arr = np.ndarray
WT = 'WarpTransformer'
PCHIP = 'PCHIP (pooled anchors)'
BASELINES = {  # method → (source file, array key)
    'Unaligned': ('base', 'unaligned'),
    'icoshift': ('base', 'icoshift'),
    'Constant shift': ('pool', 'constant_shift'),
    'Linear warp': ('pool', 'linear'),
    PCHIP: ('pool', 'pchip_pooled'),
    'COW-TIC': ('base', 'cow_tic'),
}
ANCHOR_KEYS = ('constant_shift', 'linear', 'pchip_pooled')
MARGINS = ('COW-TIC', PCHIP, 'Constant shift', 'Linear warp')
MARGIN_METRICS = ('TIC r', 'raw r', 'raw r σ1', 'raw r σ2', 'r in range', 'r non-empty')
SHARE_METRICS = ('TIC r', 'raw r σ1')


def ms(x: Arr | list[float], fmt: str = '.3f') -> str:
    """'mean ± s.d.' of x."""
    m, s = mean_sd(x)
    return f'{m:{fmt}} ± {s:{fmt.lstrip("+")}}'


def same(x: Arr, y: Arr) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and bool(np.allclose(x, y, atol=1e-6, equal_nan=True))


@dataclass
class PairData:
    a: str
    b: str
    base: dict[str, Arr]
    pool: dict[str, Arr]
    calib: dict[int, dict[str, Arr]]
    ablate: dict[int, dict[str, Arr]]
    ceiling: dict[str, float]  # metric suffix → within-batch ceiling (empty without step 11)

    @property
    def n_a(self) -> int:
        return int(self.base['n_a'])

    @property
    def n_b(self) -> int:
        return int(self.base['n_b'])

    @property
    def label(self) -> str:
        return pair_label(self.a, self.b)

    @property
    def evaluated(self) -> bool:
        return (self.a, self.b) in EVAL_PAIRS


def _load(path) -> dict[str, Arr]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_pair(paths: Paths, a: str, b: str, seeds: list[int]) -> PairData:
    pp = paths.per_pair

    def per_seed(kind: str) -> dict[int, dict[str, Arr]]:
        return {
            s: _load(f) for s in seeds if (f := pp / f'tic_r_{kind}_{a}_{b}_seed{s}.npz').exists()
        }

    ceiling: dict[str, float] = {}
    if (f := pp / 'within_batch_ceiling.npz').exists():
        z = _load(f)
        ceiling = {
            sfx: float(z[k]) for sfx in METRICS.values() if (k := f'{a}_{b}_ceiling{sfx}') in z
        }
    return PairData(
        a,
        b,
        _load(pp / f'tic_r_baselines_{a}_{b}.npz'),
        _load(pp / f'tic_r_pchip_pooled_{a}_{b}.npz'),
        per_seed('calib'),
        per_seed('ablate'),
        ceiling,
    )


def checks(d: PairData) -> dict[str, bool]:
    cal = list(d.calib.values())
    sfxs = list(METRICS.values())
    out = {
        'unaligned (06) identical across seeds and = (05) = (07), every metric': all(
            same(c['unaligned' + s], d.base['unaligned' + s])
            and same(d.pool['unaligned' + s], d.base['unaligned' + s])
            for c in cal
            for s in sfxs
        ),
    }
    for key in ANCHOR_KEYS:
        out[f'{key} (06) identical across seeds and = (07), every metric'] = all(
            same(c[key + s], d.pool[key + s]) for c in cal for s in sfxs
        )
    out['anchors identical across seeds (06, 08) and = (07)'] = all(
        same(c['anchors'], d.pool['anchors']) for c in [*cal, *d.ablate.values()]
    )
    return out


@dataclass
class PairSummary:
    data: PairData
    checks: dict[str, bool]

    def wt(self, sfx: str = '') -> Arr:
        """Per seed mean of the WarpTransformer's metric sfx."""
        return np.array([np.nanmean(c['warptransformer' + sfx]) for c in self.data.calib.values()])

    def ref(self, method: str, sfx: str = '') -> float:
        src, key = BASELINES[method]
        return float(np.nanmean(getattr(self.data, src)[key + sfx]))

    def value(self, method: str, sfx: str = '') -> float:
        """Mean over sample pairs (and over seeds for the WarpTransformer)."""
        return float(self.wt(sfx).mean()) if method == WT else self.ref(method, sfx)

    def margin(self, method: str, sfx: str = '') -> Arr:
        return self.wt(sfx) - self.ref(method, sfx)

    def wins_over_cow(self, sfx: str = '') -> list[int]:
        d = self.data
        cow = per_a_mean(d.base['cow_tic' + sfx], d.n_a, d.n_b)
        return [
            int((per_a_mean(c['warptransformer' + sfx], d.n_a, d.n_b) > cow).sum())
            for c in d.calib.values()
        ]

    def gain(self, sfx: str = '') -> Arr | None:
        if not self.data.ablate:
            return None
        return np.array(
            [
                np.nanmean(z['pretrained' + sfx]) - np.nanmean(z['random_init' + sfx])
                for z in self.data.ablate.values()
            ]
        )

    def scalars(self, method: str) -> tuple[float | str, str, str]:
        """(study silhouette, warp magnitude, outside) of a method."""
        d = self.data
        if method == WT:
            cal = list(d.calib.values())
            return (
                ms([float(c['sil'][-1]) for c in cal]),
                ms([float(c['wt_warp_mag'][2]) for c in cal]) + ' (B)',
                f'{100 * np.mean([float(c["outside"][-1]) for c in cal]):.1f}%',
            )
        src, key = BASELINES[method]
        if src == 'pool':
            return (
                float(d.pool[f'{key}_sil']),
                f'{float(d.pool[f"{key}_warp_mag"]):.3f}',
                f'{100 * float(d.pool[f"{key}_outside"]):.1f}%',
            )
        k = ('unaligned', 'cow_tic', 'icoshift').index(key)
        mag = 0.0 if k == 0 else float(d.base['warp_mag'][k - 1])
        return float(d.base['sil'][k]), f'{mag:.3f}', f'{100 * float(d.base["outside"][k]):.1f}%'


def methods() -> list[str]:
    return [*BASELINES, WT]


def format_pair(s: PairSummary) -> list[str]:
    d, seeds = s.data, list(s.data.calib)
    u = s.ref('Unaligned')
    lines = [
        '',
        f'{d.label}   n = {d.n_a}×{d.n_b}   calibration seeds {seeds}   ablation seeds '
        f'{list(d.ablate)}' + ('' if d.evaluated else '   (excluded pair)'),
        *(f'  check: {k}: {"OK" if ok else "FAILED"}' for k, ok in s.checks.items()),
        '  Every TIC metric (mean over sample pairs; WarpTransformer mean ± s.d. over seeds):',
    ]
    rows: list[list[object]] = [
        [m, *(s.ref(m, sfx) for sfx in METRICS.values())] for m in BASELINES
    ]
    rows.append([WT, *(ms(s.wt(sfx)) for sfx in METRICS.values())])
    lines += table(['Method', *METRICS], rows, indent='    ')
    lines += [
        '  Δ unaligned (normalised TIC r): '
        + '  '.join(f'{m} {s.value(m) - u:+.3f}' for m in methods()),
        '  Coverage, silhouette and warp magnitude (min; WarpTransformer: batch-B samples):',
        *table(
            ['Method', 'Study sil', 'Warp mag', 'outside'],
            [[m, *s.scalars(m)] for m in methods()],
            indent='    ',
        ),
        '  Ranking (highest first):',
        *(
            f'    {label}: '
            + ' > '.join(
                f'{m} {s.value(m, sfx):.3f}'
                for m in sorted(methods(), key=lambda m: -s.value(m, sfx))
            )
            for label, sfx in METRICS.items()
        ),
        '  WT − baseline, mean ± s.d. over seeds (seeds > 0):',
        *table(
            ['Margin', *MARGIN_METRICS],
            [
                [
                    f'WT − {m}',
                    *(
                        f'{ms(s.margin(m, METRICS[k]), "+.4f")} '
                        f'({(s.margin(m, METRICS[k]) > 0).sum()}/{len(seeds)})'
                        for k in MARGIN_METRICS
                    ),
                ]
                for m in MARGINS
            ],
            indent='    ',
        ),
    ]
    if d.ceiling:

        def share(m: str, k: str) -> str:
            sfx = METRICS[k]
            u_k = s.ref('Unaligned', sfx)
            return f'{(s.value(m, sfx) - u_k) / (d.ceiling[sfx] - u_k):.2f}'

        lines += [
            '  Share of the gap to the within-batch ceiling closed, (r − unaligned) / '
            '(ceiling − unaligned):',
            *table(
                ['Method', *(f'{k} (ceiling {d.ceiling[METRICS[k]]:.3f})' for k in SHARE_METRICS)],
                [[m, *(share(m, k) for k in SHARE_METRICS)] for m in methods()],
                indent='    ',
            ),
        ]
    lines += [
        '  WarpTransformer per seed (normalised / raw / raw σ1): '
        + '  '.join(
            f's{k} {x:.3f}/{y:.3f}/{z:.3f}'
            for k, x, y, z in zip(seeds, s.wt(), s.wt('_raw'), s.wt('_raw_s1'), strict=True)
        ),
        f'  Batch-A samples (of {d.n_a}) where WT beats COW-TIC: normalised '
        f'{s.wins_over_cow()}, raw {s.wins_over_cow("_raw")}, raw σ1 '
        f'{s.wins_over_cow("_raw_s1")}',
    ]
    mags = {
        k: np.array([c[k] for c in d.calib.values()]) for k in ('wt_warp_mag_pre', 'wt_warp_mag')
    }
    lines += table(
        ['WT warp magnitude (min)', 'all', 'A', 'B'],
        [
            [label, *(ms(m[:, i]) for i in range(3))]
            for label, m in (
                ('before calibration', mags['wt_warp_mag_pre']),
                ('after calibration', mags['wt_warp_mag']),
            )
        ],
    )
    lines += format_ablation(s)
    return lines


def format_ablation(s: PairSummary) -> list[str]:
    abl = list(s.data.ablate.values())
    if not abl:
        return ['  Pretraining ablation: no step-08 files.']
    sfxs = ('', '_raw', '_raw_s1')
    arm = {
        k + sfx: np.array([np.nanmean(z[k + sfx]) for z in abl])
        for k in ('pretrained', 'random_init')
        for sfx in sfxs
    }
    pre_w = np.array([z['warp_mag_pre'] for z in abl])  # [seeds, arm, (all, A, B)]
    post = np.array([z['warp_mag'] for z in abl])  # [seeds, arm] (all)
    devices = sorted({str(z.get('device', '?')) for z in abl})
    rows = [
        [name, *(ms(arm[key + sfx]) for sfx in sfxs), ms(pre_w[:, k, 0]), ms(post[:, k])]
        for k, (name, key) in enumerate(
            (('pretrained', 'pretrained'), ('random init', 'random_init'))
        )
    ]
    lines = [
        f'  Pretraining ablation (step 08, device {", ".join(devices)}):',
        *table(
            ['Arm', 'TIC r', 'raw r', 'raw r σ1', 'warp before (all)', 'warp after (all)'],
            rows,
            indent='    ',
        ),
    ]
    for label, sfx in (('', ''), (' (raw)', '_raw'), (' (raw σ1)', '_raw_s1')):
        gain = s.gain(sfx)
        assert gain is not None
        u = s.ref('Unaligned', sfx)
        share = (arm['random_init' + sfx] - u) / (arm['pretrained' + sfx] - u)
        lines.append(
            f'    gain per seed{label}: {np.round(gain, 4).tolist()}   mean {ms(gain, "+.4f")}   '
            f'seeds > 0: {(gain > 0).sum()}/{len(gain)}   share random init recovers: {ms(share)}'
        )
    return lines


def summarise(paths: Paths, seeds: list[int]) -> tuple[list[str], list[str]]:
    """(report lines, problems)."""
    lines = [title(__doc__), '', f'Seeds requested: {seeds}. s.d. over the seeds found (ddof 1).']
    problems: list[str] = []
    overview: dict[str, list[list[object]]] = {k: [] for k in ('TIC r', 'raw r σ1', 'r in range')}
    for a, b in EVAL_PAIRS + EXCLUDED_PAIRS:
        d = load_pair(paths, a, b, seeds)
        if not d.calib:
            if d.evaluated:
                problems.append(f'{d.label}: no step-06 files')
            continue
        s = PairSummary(d, checks(d))
        problems += [f'{d.label}: {k} FAILED' for k, ok in s.checks.items() if not ok]
        lines += format_pair(s)
        for k, rows in overview.items():
            sfx = METRICS[k]
            gain = s.gain(sfx)
            rows.append(
                [
                    d.label + ('' if d.evaluated else ' (excl.)'),
                    ms(s.wt(sfx)),
                    *(ms(s.margin(m, sfx), '+.4f') for m in MARGINS),
                    ms(gain, '+.4f') if gain is not None else None,
                ]
            )
    for k, rows in overview.items():
        lines += [
            '',
            f'Overview ({k}; mean ± s.d. over seeds)',
            *table(
                ['Pair', WT, *(f'WT − {m}' for m in MARGINS), 'pretraining gain'],
                rows,
            ),
        ]
    lines += ['', 'Checks: ' + ('all passed' if not problems else '; '.join(problems))]
    return lines, problems


def run(paths: Paths, seeds: list[int] | None = None) -> list[str]:
    lines, problems = summarise(paths, seeds if seeds is not None else [0, 1, 2, 3, 4])
    write_report(paths.results / 'seed_summary.txt', lines)
    return problems


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    setup_logging()
    if run(Paths(args.root), args.seeds):
        raise SystemExit(1)
