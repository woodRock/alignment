"""Step 13 — one summary over calibration seeds for the paper's tables.

Reads, for the four evaluated pairs, the arrays of steps 05 (icoshift, COW-TIC), 07 (PCHIP),
06 (WarpTransformer, per seed) and 08 (pretraining ablation, per seed), and reports per pair:
  - consistency checks: the deterministic arrays every seed's file carries (unaligned, PCHIP,
    anchors) are identical across seeds and match steps 05 and 07
  - TIC r (normalised and raw) of every method; the WarpTransformer as mean ± s.d., min, max
  - WT − COW-TIC and WT − PCHIP over seeds, and the number of seeds at which each is positive
  - per batch-A sample (r averaged over B_j): how many the WT beats COW-TIC for, per seed
  - warp magnitudes before and after calibration
  - the ablation: both arms' TIC r and warps (before and after), the pretraining gain per seed,
    and the share of the improvement over unaligned that random init recovers
s.d. is the sample s.d. over seeds (ddof = 1).

Output: results/seed_summary.txt (exit status 1 if a check fails)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EVAL_PAIRS, Paths, pair_label
from ..metrics import mean_sd, per_a_mean
from ._common import parser, setup_logging, table, title, write_report

Arr = np.ndarray


def ms(x: Arr | list[float], fmt: str = '.3f') -> str:
    """'mean ± s.d.' of x."""
    m, s = mean_sd(x)
    return f'{m:{fmt}} ± {s:{fmt.lstrip("+")}}'


def same(x: Arr, y: Arr) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and bool(np.allclose(x, y, atol=1e-6))


@dataclass
class PairData:
    a: str
    b: str
    base: dict[str, Arr]
    pool: dict[str, Arr]
    calib: dict[int, dict[str, Arr]]
    ablate: dict[int, dict[str, Arr]]

    @property
    def n_a(self) -> int:
        return int(self.base['n_a'])

    @property
    def n_b(self) -> int:
        return int(self.base['n_b'])

    @property
    def label(self) -> str:
        return pair_label(self.a, self.b)


def _load(path) -> dict[str, Arr]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_pair(paths: Paths, a: str, b: str, seeds: list[int]) -> PairData:
    pp = paths.per_pair

    def per_seed(kind: str) -> dict[int, dict[str, Arr]]:
        return {
            s: _load(f) for s in seeds if (f := pp / f'tic_r_{kind}_{a}_{b}_seed{s}.npz').exists()
        }

    return PairData(
        a,
        b,
        _load(pp / f'tic_r_baselines_{a}_{b}.npz'),
        _load(pp / f'tic_r_pchip_pooled_{a}_{b}.npz'),
        per_seed('calib'),
        per_seed('ablate'),
    )


def checks(d: PairData) -> dict[str, bool]:
    cal = list(d.calib.values())
    s0 = cal[0]
    return {
        'unaligned (06) identical across seeds': all(
            same(c['unaligned'], s0['unaligned']) for c in cal
        ),
        'unaligned (06) = unaligned (05)': same(s0['unaligned'], d.base['unaligned']),
        'unaligned raw (06) identical across seeds and = (05) = (07)': all(
            same(c['unaligned_raw'], d.base['unaligned_raw']) for c in cal
        )
        and same(d.base['unaligned_raw'], d.pool['unaligned_raw']),
        'PCHIP (06) identical across seeds and = (07)': all(
            same(c['pchip_pooled'], d.pool['pchip_pooled']) for c in cal
        ),
        'PCHIP raw (06) identical across seeds and = (07)': all(
            same(c['pchip_pooled_raw'], d.pool['pchip_pooled_raw']) for c in cal
        ),
        'anchors identical across seeds (06, 08) and = (07)': all(
            same(c['anchors'], d.pool['anchors']) for c in [*cal, *d.ablate.values()]
        ),
    }


@dataclass
class PairSummary:
    data: PairData
    checks: dict[str, bool]
    wt: Arr = field(init=False)
    wt_raw: Arr = field(init=False)

    def __post_init__(self) -> None:
        self.wt = np.array([c['warptransformer'].mean() for c in self.data.calib.values()])
        self.wt_raw = np.array([c['warptransformer_raw'].mean() for c in self.data.calib.values()])

    def ref(self, key: str, raw: bool = False) -> float:
        src = self.data.pool if key == 'pchip_pooled' else self.data.base
        return float(src[key + ('_raw' if raw else '')].mean())

    def margin(self, key: str, raw: bool = False) -> Arr:
        return (self.wt_raw if raw else self.wt) - self.ref(key, raw)

    def wins_over_cow(self, raw: bool = False) -> list[int]:
        sfx = '_raw' if raw else ''
        cow = per_a_mean(self.data.base['cow_tic' + sfx], self.data.n_a, self.data.n_b)
        return [
            int((per_a_mean(c['warptransformer' + sfx], self.data.n_a, self.data.n_b) > cow).sum())
            for c in self.data.calib.values()
        ]

    def gain(self, raw: bool = False) -> Arr | None:
        if not self.data.ablate:
            return None
        sfx = '_raw' if raw else ''
        return np.array(
            [
                z['pretrained' + sfx].mean() - z['random_init' + sfx].mean()
                for z in self.data.ablate.values()
            ]
        )


def format_pair(s: PairSummary) -> list[str]:
    d, seeds = s.data, list(s.data.calib)
    u, u_raw = s.ref('unaligned'), s.ref('unaligned', raw=True)
    lines = [
        '',
        f'{d.label}   n = {d.n_a}×{d.n_b}   calibration seeds {seeds}   ablation seeds '
        f'{list(d.ablate)}',
        *(f'  check: {k}: {"OK" if ok else "FAILED"}' for k, ok in s.checks.items()),
    ]
    rows: list[list[object]] = [
        [name, s.ref(key), s.ref(key) - u, s.ref(key, True), s.ref(key, True) - u_raw]
        for name, key in (
            ('Unaligned', 'unaligned'),
            ('icoshift', 'icoshift'),
            ('PCHIP (pooled anchors)', 'pchip_pooled'),
            ('COW-TIC', 'cow_tic'),
        )
    ]
    rows.append(
        ['WarpTransformer', ms(s.wt), s.wt.mean() - u, ms(s.wt_raw), s.wt_raw.mean() - u_raw]
    )
    lines += table(['Method', 'TIC r', 'Δ unal', 'raw r', 'Δ unal'], rows)
    lines += [
        '  WarpTransformer per seed (normalised/raw): '
        + '  '.join(f's{k} {x:.3f}/{y:.3f}' for k, x, y in zip(seeds, s.wt, s.wt_raw, strict=True))
    ]
    lines += table(
        ['Margin', 'normalised', 'seeds > 0', 'raw', 'seeds > 0'],
        [
            [
                name,
                ms(s.margin(key), '+.4f'),
                f'{(s.margin(key) > 0).sum()}/{len(seeds)}',
                ms(s.margin(key, True), '+.4f'),
                f'{(s.margin(key, True) > 0).sum()}/{len(seeds)}',
            ]
            for name, key in (('WT − COW-TIC', 'cow_tic'), ('WT − PCHIP', 'pchip_pooled'))
        ],
    )
    lines += [
        f'  Batch-A samples (of {d.n_a}) where WT beats COW-TIC: normalised '
        f'{s.wins_over_cow()}, raw {s.wins_over_cow(raw=True)}'
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
    lines += format_ablation(s, u, u_raw)
    return lines


def format_ablation(s: PairSummary, u: float, u_raw: float) -> list[str]:
    abl = list(s.data.ablate.values())
    gain, gain_raw = s.gain(), s.gain(raw=True)
    if gain is None or gain_raw is None:
        return ['  Pretraining ablation: no step-08 files.']
    arm = {
        k: np.array([z[k].mean() for z in abl])
        for k in ('pretrained', 'random_init', 'pretrained_raw', 'random_init_raw')
    }
    pre_w = np.array([z['warp_mag_pre'] for z in abl])  # [seeds, arm, (all, A, B)]
    post = np.array([z['warp_mag'] for z in abl])  # [seeds, arm] (all)
    devices = sorted({str(z.get('device', '?')) for z in abl})
    share = (arm['random_init'] - u) / (arm['pretrained'] - u)
    share_raw = (arm['random_init_raw'] - u_raw) / (arm['pretrained_raw'] - u_raw)
    rows = [
        [name, ms(arm[key]), ms(arm[key + '_raw']), ms(pre_w[:, k, 0]), ms(post[:, k])]
        for k, (name, key) in enumerate(
            (('pretrained', 'pretrained'), ('random init', 'random_init'))
        )
    ]
    return [
        f'  Pretraining ablation (step 08, device {", ".join(devices)}):',
        *table(
            ['Arm', 'TIC r', 'raw r', 'warp before (all)', 'warp after (all)'], rows, indent='    '
        ),
        f'    gain per seed: {np.round(gain, 4).tolist()}   mean {ms(gain, "+.4f")}   '
        f'seeds > 0: {(gain > 0).sum()}/{len(gain)}',
        f'    gain per seed (raw): {np.round(gain_raw, 4).tolist()}   mean '
        f'{ms(gain_raw, "+.4f")}   seeds > 0: {(gain_raw > 0).sum()}/{len(gain_raw)}',
        f'    share of the improvement random init recovers: {ms(share)} (raw {ms(share_raw)})',
    ]


def summarise(paths: Paths, seeds: list[int]) -> tuple[list[str], list[str]]:
    """(report lines, problems)."""
    lines = [title(__doc__), '', f'Seeds requested: {seeds}. s.d. over the seeds found (ddof 1).']
    problems: list[str] = []
    overview: list[list[object]] = []
    for a, b in EVAL_PAIRS:
        d = load_pair(paths, a, b, seeds)
        if not d.calib:
            problems.append(f'{d.label}: no step-06 files')
            continue
        s = PairSummary(d, checks(d))
        problems += [f'{d.label}: {k} FAILED' for k, ok in s.checks.items() if not ok]
        lines += format_pair(s)
        gain = s.gain()
        overview.append(
            [
                d.label,
                ms(s.wt),
                ms(s.margin('cow_tic'), '+.4f'),
                ms(s.margin('pchip_pooled'), '+.4f'),
                ms(gain, '+.4f') if gain is not None else None,
            ]
        )
    lines += [
        '',
        'Overview (normalised TIC r; mean ± s.d. over seeds)',
        *table(
            ['Pair', 'WarpTransformer', 'WT − COW-TIC', 'WT − PCHIP', 'pretraining gain'], overview
        ),
        '',
        'Checks: ' + ('all passed' if not problems else '; '.join(problems)),
    ]
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
