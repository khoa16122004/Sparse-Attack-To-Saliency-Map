# Sparse-Attack-To-Saliency-Map

### Batch explanation methods

`run_batch.py --explain-method` supports `simple_gradient`,
`integrated_gradients`, `input_gradient`, `grad_cam`, `raw_attention`,
`xgrad_cam`, and `attention_grad`. The shared method dispatcher resolves
`xgrad_cam` to the existing XGrad-CAM implementation.

### Interactive Pareto comparison

```powershell
python -m pip install -r requirements.txt
python -m streamlit run evaluate_script\paretofront_compare_ui.py
```

Set GA and NSGA-II roots in the sidebar. Each root must contain
`seed/model/approach/class/sample/non_dominated_front_scores.txt`, as used by
the existing `paretofront_compare.py` CLI. Choose seed, model, strategy,
explanation, loss, epsilons, and GA weights. Duplicate runs require explicit
selection. Use **Reload files** after experiment results change.

The top-30 table suggests sample/epsilon/GA-weight combinations with the
largest NSGA-II strict Pareto coverage advantage. Filter by forward/reverse
coverage or sample name, then inspect any matching pair using the sample
selector and Previous/Next buttons. Both losses are minimized; identical points
do not dominate each other, and duplicate coordinates count once. Net advantage
is the percentage of GA points dominated by NSGA-II minus the percentage of
NSGA-II points dominated by GA, not hypervolume. Missing/invalid fronts are
reported and excluded pairwise, not scored as wins. Rankings are selected cases,
not aggregate evidence that one algorithm always wins. Loss types are never
mixed, including GA 0:1 runs with different stored objective coordinates.

The default **Paper figure** view uses a white background, serif fonts, subtle
grid, hollow colored-border markers connected in increasing objective order,
and a vertical legend inside the upper-right corner without a sample title.
Markers have white centers that mask connecting lines, so segments meet the
border without crossing the center. Axes use LaTeX
`$\mathcal{L}_{\text{attack}}$` and `$\mathcal{L}_{\text{explain}}$`.
GA fronts are labeled `GASS ($\lambda=...$)` with lambda equal to the attack
weight `w_m`, matching the scoring tables; NSGA-II is bold **Ours** in dark red.
Straight segments connect observed points without smoothing or interpolation.
Download PNG at 600 dpi or vector PDF/SVG for a paper.
Paper figures default to a square 380-pixel (3.17-inch) canvas for a single column,
with a square plotting area independent of the two axes' numerical scales,
with 4% padding around the data for a tighter view without dropping points.
Axis titles use 16-point type; both axes use sparse, evenly spaced round-number
ticks (1/2/5-based steps), rather than forcing exactly three arbitrary values.
The upper-right legend uses a 10-point font, spaced rows, and a subtle white
background; no space is reserved below the plot for a legend.
Adjust square figure size and marker size in the sidebar; use **Crop / zoom for export**
to set exact axis limits for both preview and downloads.
The optional **Interactive** view supports box/scroll zoom, double-click reset,
and standalone HTML export. Optional min-max scaling only affects coordinates
and axis labels, not dominance scores. Export the filtered ranking as CSV.
The existing batch PNG/history-curve CLI remains available.

### Check experiment completeness

Audit runs without recomputing scores or reading images:
By default, only run folders are checked; no sample folders or summary files
are read. Use `--check-samples` for the slower per-sample validation.

```powershell
python evaluate_script\check_run_completeness.py --root "D:\results\server_run_seed"
python evaluate_script\check_run_completeness.py --cached
python evaluate_script\check_run_completeness.py --root "D:\results\server_run_seed" --check-samples
python evaluate_script\check_run_completeness.py --root "D:\results\server_run_seed" --seed 22520692 --model resnet18 --explain-method grad_cam
```

The default expected grid contains 1,440 combinations: seeds 22520691-22520695,
models resnet18/vgg16/vit_b_32, explanation methods simple_gradient/input_gradient/
grad_cam/raw_attention, epsilons 20/50/100, GA weights 0:1/0.5:0.5/1:0,
NSGAII weights 0.5:0.5, and both margin and negative-cross-entropy losses.
Only uniform initialization is checked, matching the scoring tables.
Override the expected dimensions with `--expected-seeds`, `--expected-models`,
and `--expected-explain-methods` (space-separated values). Single-value filters
`--seed`, `--model`, `--algorithm`, `--explain-method`, and `--eps` restrict the grid.

Reports are saved as `completeness_report.txt`, `completeness_report.csv`, and
`completeness_report.json` under `--output-dir` (default:
`evaluate_script/stats_outputs`). They include every expected combination,
seed/model/explanation summaries, run paths, malformed-summary errors, and
out-of-scope runs in JSON.

`present` means the run folder exists, with sample completeness unverified.
Sample counts and score usability are null when not checked.
With `--check-samples` (or sample counts from cache),
`complete` means at least `--expected-samples 100` valid sample summaries;
`partial` means unfinished but meeting `--min-sample-ratio 0.5`;
`insufficient` means an existing run below that threshold; `missing` means no run.
If a run contains more summaries than expected, all of them must be valid to
count as complete. Sample status `ok` means processing completed, not necessarily
a successful adversarial attack. Duplicate runs are reported separately and
the best sample count is used, never added together.
Physically missing losses remain missing even for GA 0:1, with an equivalent-loss
note where applicable.

`--cached` / `--load-cached`, `--all-runs-json PATH`, or `--root PATH.json`
read an existing scoring cache without scanning disk. Cache-only reports cannot
recover runs already filtered out by scoring or distinguish mirrored equivalent
losses from actual runs; use a root scan for physical completeness. The audit
never overwrites the scoring cache or LaTeX reports.


### Experiment shell scripts

All GA/NSGAII launchers in `script_sh` run only `margin_saliency`, sequentially
over seeds 22520691, 22520692, 22520693, 22520694, and 22520695.
Each seed writes to its own output directory. Model-specific scripts preserve
their existing budgets, explanation method, and objective-weight defaults;
GA scripts do not automatically sweep all three weight settings.
The legacy `script_compare_loss_50.sh` name is retained, but it now runs only
Margin Loss. Evaluation/reporting scripts are unaffected by this seed sweep.

### Faithfulness comparison

```bash
sbatch script_sh/run_evaluate_faithfulness.sh /path/to/GA/results /path/to/NSGAII/results
```

Pass all experiment roots together (or one root containing all approaches).
The script evaluates Clean saliency and linear min-max normalized `smoothed_density` maps
on the clean RGB image for `true_label`. It compares five rows: Clean,
GA(lambda=0.0), GA(lambda=0.5), GA(lambda=1.0), and NSGAII; lambda is `w_m`.
Adversarial saliency is no longer recomputed for this report.

Each model/explainer/epsilon/loss block includes all available seeds and approaches.
After each block, the evaluator updates CSV/JSON/LaTeX tables and saves separate
Insertion/Deletion curves as PNG and PDF, together with `curves.json`.
Outputs live under `faithfulness_aggregates/comparison_<configuration-hash>/`,
separated by strategy, evaluation protocol, and density metadata, not algorithm
or weights. Tables have epsilon 20/50/100 columns, each with Margin Loss and
Log-Likelihood Ins/Del/IMD values. Missing cells are `--`.
Density no longer applies gamma; legacy gamma metadata is ignored for grouping.
For old density archives, evaluation reconstructs the linear map from
`smoothed_density`, not the potentially gamma-adjusted `normalized_density`.
Missing `smoothed_density` is an error requiring density regeneration.
Insertion blur remains kernel size 11, sigma 5; density grid and smoothing
settings are unchanged. Rerun the evaluator with `--replace` to replace scores
from the old density protocol. Old comparison output folders are not deleted;
use a fresh `OUTPUT_DIR` to avoid confusing them with the new report.
Clean is counted once per seed/class/sample within each block, not once per
approach. During recomputation, its explanation and Ins/Del scores are computed
once per sample in the block and shared across approaches after checking the
decoded clean image hash and true label. Different images or labels raise errors;
the density map is still evaluated separately for every approach.
Old independently computed clean curves can disagree; rerun the evaluator with
`--replace` to regenerate these using the shared baseline. Standalone aggregation
still rejects inconsistent clean curves and duplicate approach samples.
Blocks missing approaches or with different sample cohorts are marked `PARTIAL`
in the figures and JSON; they are not complete matched comparisons.

The SLURM wrapper always passes `--replace`, recomputing and overwriting
`faithfulness.json` evaluations rather than reusing cached scores. This does not
rerun the attacks. When invoking the Python evaluator directly, existing successful
evaluations are reused unless `--replace` is supplied. Mismatched cached context
or curve settings require `--replace`.
To rebuild reports only, run:

```bash
python evaluate_script/run_aggregate_faithfulness.py --input-dirs /path/to/GA/results /path/to/NSGAII/results --output-dir faithfulness_aggregates
```

The SLURM wrapper accepts `DEVICE`, `STEP`, `BATCH_SIZE`, and `OUTPUT_DIR`
environment overrides. Without positional roots it uses `INPUT_DIR`
(default `offical/server_run_seed`).

### Quick Start


```sh
python run_sample.py \
	--image test_imgs/cat.png \
	--model vgg16 \
	--algorithm weighted_sum_ga \
	--fitness-function margin_saliency \
	--w-margin 0.5 \  
	--w-saliency 0.5 \
	--operator-strategy saliency_guided \
	--eps 50 \
	--iterations 50 \
	--pop-size 20 \
	--output outputs/sample_vgg16/adv.png \
	--clean-image-output outputs/sample_vgg16/clean.png \
	--clean-map-output outputs/sample_vgg16/clean_map.png \
	--adv-map-output outputs/sample_vgg16/adv_map.png \
	--save-history-chart \
	--history-chart-output outputs/sample_vgg16/history.png
```



The command above creates these main files:

- `outputs/sample_vgg16/clean.png`: clean image after resize/crop.
- `outputs/sample_vgg16/adv.png`: adversarial image.
- `outputs/sample_vgg16/clean_map.png`: saliency map of the clean image.
- `outputs/sample_vgg16/adv_map.png`: saliency map of the adversarial image.
- `outputs/sample_vgg16/history_margin.png` and `outputs/sample_vgg16/history_saliency.png`: optimization history charts.

The terminal also prints an `Attack summary`, including clean/adv predictions, `l0_distance`, `margin_loss`, `saliency_loss`, and more.
