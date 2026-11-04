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

### Experiment models

The standard CNN models are `resnet18`, `vgg16`, and `densenet121`.
The Transformer models are `vit_b_32`, `vit_b_16`, and Swin V2 Tiny
(`swin_v2_t` in torchvision; `swin_v2` is also accepted by the shared loader
and is the canonical reporting name). Attack and faithfulness tables use
this same six-model order. Evaluation loads the model recorded in each run;
it does not substitute ResNet-50 or EfficientNet for another architecture.
Existing generated tables must be regenerated to update their model headings;
do not rename old ResNet-50/EfficientNet experiment data as ResNet-18/DenseNet.

### Attack LaTeX tables from multiple roots

```sh
python -m evaluate_latex_generate.generate_tables \
    --attack-input-dirs path/to/preferred_runs path/to/reruns \
    --duplicate-runs first --output-dir result_latexs --no-ttest
```

Duplicate attack runs are rejected by default. Use `--duplicate-runs first`
only for copies or reruns: the first run encountered for each
`(model, explainer, approach, seed, strategy, eps, loss_type)` is kept.
Roots are processed in command-line order; within each root, the loader's
run order determines priority. Each discarded run is reported on stderr with
both paths and recorded under `duplicate_runs` in the table's `statistics.json`.
The same option works with `--all-runs-json file_a.json file_b.json`.
It does not merge split sample sets and cannot be used with faithfulness inputs.

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

# CPU only: re-aggregate existing faithfulness.json files and re-render tables.
# Disable IoU when SAM3 masks / original images / saved attack maps are not available.
RUN_IOU=0 bash script_sh/run_aggregate_faithfulness.sh offical_faithfulness/server_run_seed
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

### SAM3 pseudo-GT masks and localization IoU

SAM3 text-prompt masks are **pseudo-ground-truth**, not human annotations.
The GPU mask-generation job is separate from the CPU-only Bash reporting script.
Activate an environment with the notebook's SAM3 installation, compatible
PyTorch/torchvision and the SAM3 checkpoint access before submission. SAM3 is
an optional dependency; it is not needed in the CPU reporting environment.
The notebook used SAM3 revision `44ef224799e7b32e017855823883909d38ffb30f`
and PyTorch 2.10.0. No script installs packages automatically.

#### 1. Generate masks once (GPU / sbatch)

[The SAM3 generator](evaluate_script/generate_imagenet_masks.py) takes an
ImageNet root directly containing class folders. Default text prompts are the
first class names in [synset_words.txt](synset_words.txt). To customize them,
pass `PROMPTS_JSON` pointing to a JSON object such as
`{"n01440764": "tench fish", "n02123045": "tabby cat"}`.
The confidence threshold defaults to **0**. Exactly one candidate is selected:
the nonempty mask with the highest returned SAM3 confidence score (first candidate
on ties), not a union of instances. Candidates with no foreground pixels above
the mask threshold of 0.5 are excluded before ranking. Confidence, original
candidate index and total candidate count are saved per image. No candidates or
all-empty candidates cause an explicit error; the generator does not fabricate
a pseudo-GT. The nonempty selection policy is recorded in the manifest.
Both shell runners default to `/datastore/quanphm/datasets/ImageNet/val`;
`imagenet_mask` is created relative to the repository root, not inside the
shared ImageNet directory. Mask generation processes every image under the
input root (the full validation set when using this default).
CUDA inference wraps both image encoding and text prompting in BF16 autocast
and requires a BF16-capable GPU; model weights are not manually converted.
The selected precision is recorded in the mask manifest.
To bound candidate-mask memory for large source images, the longest image side
passed to SAM3 is capped at 1024 pixels, preserving aspect ratio and never
upscaling. Only the selected binary mask is restored to the original resolution
on CPU using nearest-neighbor interpolation. This can change segmentation
results; the resize policy and per-image inference dimensions are recorded in
the manifest. GPU tensors from each image are released before the next inference.
Override the cap with `SAM_MAX_IMAGE_SIDE` (or `--max-image-side` for Python).
Lowering it can reduce candidate-mask memory, but does not reduce model weights
or guarantee that inference fits on a busy GPU.

```sh
conda activate Sparse_Attack
sbatch script_sh/run_generate_imagenet_masks.sh /path/to/imagenet imagenet_mask

# Optional local checkpoint and class-specific prompts:
SAM_CHECKPOINT=/path/to/sam3.pt PROMPTS_JSON=class_prompts.json \
    sbatch script_sh/run_generate_imagenet_masks.sh /path/to/imagenet imagenet_mask

# Regenerate existing masks explicitly:
REPLACE=1 sbatch script_sh/run_generate_imagenet_masks.sh /path/to/imagenet imagenet_mask

# Smaller inference images if GPU memory is limited:
SAM_MAX_IMAGE_SIDE=768 REPLACE=1 \
    sbatch script_sh/run_generate_imagenet_masks.sh /path/to/imagenet imagenet_mask
```

For example, `imagenet/n01440764/example.JPEG` becomes
`imagenet_mask/n01440764/example.png`. Subdirectories and image stems are
preserved; the extension becomes PNG to avoid lossy binary masks. Masks have
the **original image resolution**, grayscale background 0 and foreground 255.
The generator rejects duplicate output stems and writes a completed
`mask_manifest.json` with prompts, hashes and selected confidence scores. An interrupted
job leaves an incomplete manifest which the evaluator rejects. Review the
prompt if no valid mask is returned. `SAM_CONFIDENCE` defaults to 0.
Regenerate old masks with `REPLACE=1` to apply the nonempty selection protocol;
this is also required to rerun a partially completed generation job.
Even a nonempty original-resolution mask can become empty after model
center-crop; lowering confidence cannot guarantee foreground survives cropping.

#### 2. IoU plus existing Ins / Del / IMD reports (CPU, no model inference)

[The CPU runner](script_sh/run_aggregate_faithfulness.sh) now runs
[localization IoU](evaluate_script/run_localization_iou.py), then aggregates
existing insertion/deletion/IMD scores and renders their existing tables:

```sh
IMAGENET_ROOT=/path/to/imagenet MASK_ROOT=imagenet_mask IOU_THRESHOLD=0.5 \
    bash script_sh/run_aggregate_faithfulness.sh path/to/seed_runs

# Multiple roots are combined into the same reports (quote paths containing spaces).
bash script_sh/run_aggregate_faithfulness.sh path/to/seed_1 path/to/seed_2

# IoU only, without requiring precomputed faithfulness.json:
python -m evaluate_script.run_localization_iou \
    --input-dirs path/to/seed_runs \
    --imagenet-root /path/to/imagenet --mask-root imagenet_mask \
    --threshold 0.5 --output-dir result_latexs
```

With no positional roots, the CPU runner uses its `INPUT_DIRS` array;
`INPUT_DIR` overrides that array when set. Progress bars show file discovery,
JSON loading, localization IoU, faithfulness aggregation/curves, and table output.
Discovery bars count matching files without a known total; they are not percentages
of all directories scanned.

Input run roots must retain the successful sample summaries, `clean.png`
and `perturbation_density.pt`, in the existing `strategy-*/class/image_stem/`
layout. The selected Clean baseline must also retain `clean_map.png` (or
`clean_map.pt`); other runs' Clean maps are not read. The combined runner additionally
requires the precomputed `faithfulness.json` files. It **does not calculate
new Ins/Del/IMD model predictions**, and does not import SAM3 or load a model.
The CPU environment needs NumPy, Pillow, PyTorch, torchvision and SciPy.

IoU protocol:

- Compare Clean saliency and each GASS/Ours perturbation-density map with the
  same class/sample pseudo-GT. Independently min-max normalize each map to
  `[0, 1]`, then select pixels **>= 0.5** by default. Constant maps normalize
  to zero and therefore produce empty predictions at this threshold.
  Set `IOU_THRESHOLD` in the shell runner or pass `--threshold` to change it.
  For the legacy fixed-area ranking metric, explicitly set `TOP_K_PERCENT=80`
  or pass `--top-k-percent 80` instead: exactly `ceil(H * W * k / 100)`
  pixels, descending stable row-major tie order, including constant maps.
  Thresholding affects localization only; faithfulness Ins/Del/IMD continues
  to use the full saliency ranking without binarization.
- Use model-default resize and center-crop coordinates, nearest-neighbor
  interpolation for binary GT. Verify the transformed source RGB image against
  saved `clean.png` before scoring; wrong images/transforms fail explicitly.
- For each (model, explainer, class, image), read and score Clean only once
  from the first seed with a successful sample surviving the CLI filters.
  Numeric seeds sort numerically; nonnumeric seeds follow in lexical order.
  Within that seed, choose the first sample by sorted full path, independently
  of input-root argument order. Reuse this baseline across all seeds and
  approaches, not each run's own Clean map. Log the selected source and save
  `clean_baseline_dir`/`clean_baseline_seed` in the exact records (also retained
  in combined-table provenance). Every run must still have matching Clean RGB
  and GT. This is a shared-baseline protocol, not independent Clean trials;
  identical sample sets therefore give zero Clean variation across seeds.
- New batch attacks save the unquantized `clean_map.pt`. Older runs use the
  inverse of the project's RGB heatmap ramp, **not grayscale luminance**;
  quantization ties are recorded in the protocol. Density uses the saved
  `smoothed_density` tensor, not a colored overlay or legacy gamma mapping.
- Empty GT (including foreground removed by center-crop) is an error by default.
  To deliberately retain these as IoU=0, set `EMPTY_GT=zero` in the shell runner
  or pass `--empty-gt zero`; these cases are explicitly logged.
- Compute binary intersection/union per sample, average within each seed,
  then report equally weighted mean ± sample SD across seeds. One seed has
  undefined SD (`--`). Clean is deduplicated across approaches; duplicate runs
  or inconsistent Clean/GT inputs are rejected.

With `RUN_IOU=1` (the default), the CPU runner also joins the exact per-sample
IoU records into the faithfulness table: **Ins / Del / IMD / IoU** for each
model and Clean/GASS/Ours row. All four metrics use the same samples and seeds;
missing/extra samples or mismatched experiment contexts cause an error.
IoU uses the existing mean/SD, best-value bolding and optional paired t-test
logic (higher is better). The combined table records the localization protocol,
including threshold/normalization (or legacy top-k) and GT manifest hash,
in its configuration. Regenerate localization records to apply the new
threshold; joining saved records does not recompute IoU.

IoU-only outputs go to `result_latexs/localization_<hash>/`:
`table.tex` (Clean/GASS/Ours, best means bold), `samples.csv` (per-sample IoU),
and `statistics.json` (per-seed values, configuration, provenance and counts).
Different eps/loss/strategy, GT manifests, map encodings or protocols are kept
in separate tables. Combined Ins/Del/IMD/IoU tables are in `faithfulness_<hash>/`;
`RUN_IOU=0` instead produces the original three-metric faithfulness tables.
The exact records used to join are saved as
`faithfulness_aggregates/localization_records.json` (under `OUTPUT_DIR`).
The runner forwards `EPS`, `LOSS_TYPE`, `STRATEGY`, `PRECISION` and `TABLE_DIR`
to IoU reporting; `RUN_IOU=0` preserves the old faithfulness-only workflow.

#### 3. Combine already saved per-sample scores only

If `faithfulness.json` and `localization_records.json` already exist, use
[the join-only runner](script_sh/run_combined_faithfulness_localization.sh).
It does not regenerate masks, calculate IoU, or run model inference:

```sh
LOCALIZATION_RECORDS=faithfulness_aggregates/localization_records.json \
    bash script_sh/run_combined_faithfulness_localization.sh \
    offical_faithfulness/server_run_seed
```

Pass one or more run roots containing per-sample `faithfulness.json` files.
To run without arguments, edit the `INPUT_DIRS` array in the join-only runner,
adding one quoted root per line, then run:

```sh
bash script_sh/run_combined_faithfulness_localization.sh
```

Command-line roots override the configured array; when no arguments are given,
a non-empty `INPUT_DIR` environment variable overrides it with a single root.
The array initially contains `offical_faithfulness/server_run_seed`.
`LOCALIZATION_RECORDS` must cover exactly the samples in the selected roots
after filtering. Output is
`result_latexs/faithfulness_<hash>/table.tex` and `statistics.json`, with
Ins / Del / IMD / IoU columns. The join-only runner defaults to three decimal
places, SD as a lower-right subscript (`$0.277_{0.001}$`), and bold best means
without blue t-test highlighting (`USE_TTEST=0`). Set `USE_TTEST=1` to re-enable
the significance tests and coloring. Existing outputs for the same configuration
are overwritten. `TABLE_DIR`, `PRECISION`, `USE_TTEST`, `TTEST_ALPHA`,
`TTEST_CORRECTION`, `EPS`, `LOSS_TYPE`, and `STRATEGY` use the same conventions
as the CPU aggregation runner.

The join checks exact sample paths and contexts, and rejects duplicate,
missing, or extra samples instead of merging unrelated scores. Use the same
run roots and filters that produced the localization records. If results
were moved from another machine, the `sample_dir` paths in both inputs must
resolve to the same samples. A localization `table.tex` alone is not enough:
the exact records JSON is required.

Focused tests are provided for manual execution:

```sh
python -m unittest discover -s tests -p test_localization_iou.py
python -m unittest discover -s tests -p test_latex_tables.py
```
