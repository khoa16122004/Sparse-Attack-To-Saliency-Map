# Sparse-Attack-To-Saliency-Map


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

### Faithfulness scoring

Activate the project's Python environment with CUDA-enabled PyTorch, then run
[the faithfulness scoring script](script_sh/run_score_faithfulness.sh) with Bash
on the GPU machine:

```sh
conda activate Sparse_Attack
bash script_sh/run_score_faithfulness.sh offical/server_run_seed
```

Multiple input roots can be passed as separate arguments. The script first
evaluates each sample on CUDA (insertion/deletion AUC and IMD for clean and
perturbation maps), then aggregates the saved `faithfulness.json` files into
metrics and Ins/Del curves. Finally, it renders paper tables with the fixed
[faithfulness template](evaluate_latex_generate/faithfulness.text): Clean,
GASS ($\lambda=0, 0.5, 1$) and Ours, with Ins/Del/IMD as mean ± SD across seeds.
Like the ASR tables, each `(eps, loss_type, strategy)` configuration is written
to its own `result_latexs/faithfulness_<hash>/table.tex`; see that folder's
`statistics.json` for its configuration. Existing compatible sample evaluations
are reused; stale context/protocol settings cause an error rather than being
silently reused. Relative paths are resolved from the repository root.

```sh
# Recompute every sample and customize the output directory/batch size.
REPLACE=1 BATCH_SIZE=8 OUTPUT_DIR=faithfulness_scores \
    bash script_sh/run_score_faithfulness.sh path/to/seed_runs path/to/more_runs

# Render only selected table configurations.
EPS="20 50" LOSS_TYPE=margin_loss STRATEGY=uniform \
    bash script_sh/run_score_faithfulness.sh offical/server_run_seed

# Alternatively submit to the existing UIT Slurm GPU partition.
sbatch script_sh/run_score_faithfulness.sh offical/server_run_seed
```

Environment overrides: `PYTHON` (executable, default `python`), `INPUT_DIR`
(used only when no positional roots are provided), `OUTPUT_DIR` for metrics
and curves (default `faithfulness_aggregates`), `TABLE_DIR` for LaTeX tables
(default `result_latexs`), `STEP` (224 pixels), `BATCH_SIZE` (16),
`KERNEL_SIZE` (11), `KERNEL_SIGMA` (5), and `REPLACE` (0 or 1). Optional
table filters are `EPS` (space-separated), `LOSS_TYPE` and `STRATEGY`; filters
do not limit which samples are evaluated. T-test settings are `USE_TTEST`
(1 or 0), `TTEST_ALPHA` (0.05), `TTEST_CORRECTION` (`none` or `bonferroni`),
and `PRECISION` (4 decimal places).
