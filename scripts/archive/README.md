# Archived scripts

Not used by the WarpTransformer paper. Kept for reference; they may need path fixes to run
from here (several load sibling scripts or `../src` relative to their own location).

- `01`, `02`, `07`–`12`: earlier wheat/rice cross-study work (MTBLS21 / MTBLS288).
- `04`–`06`: SimCLR spectrum-encoder pretraining (the paper's anchor matching uses raw cosine, no encoder).
- `14`, `15`: dataset-level alignment and cross-batch fine-tuning, superseded by test-time calibration (16).
- `19`: older five-panel alignment figure, replaced by `20_generate_figures.py`.
- `23`: anchor-count ablation; out of scope for the paper (mentioned only in Limitations).
- `run_gpu_spooler.sh`: GPU-server runner for an earlier run plan; its step numbers are stale.
