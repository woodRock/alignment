"""One module per pipeline step, in run order (scripts/NN_*.py call their cli()):

01 download_library   02 pretrain        03 preprocess        04 audit
05 literature_baselines   06 calibrate_pair   07 pchip_baseline   08 ablate_pretrain
09 mechanism_ablation   10 heldout_anchors   11 ceiling   12 significance
13 aggregate_seeds   14 figures

Each exposes run(paths, …) for programmatic use and cli(argv) for the command line; most also
split compute(…) → result, format_report(result) → lines, and save(result, paths).
"""
