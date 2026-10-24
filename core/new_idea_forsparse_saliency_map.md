## Evolutionary Perturbation Density Map Generation

For each input image, the evolutionary attack algorithm (GA or NSGA-II) maintains a per-image archive of non-dominated sparse perturbation solutions throughout optimization. Each candidate is evaluated using the original attack and explanation objectives, and the archive is updated to retain non-dominated, unique solutions.

After optimization, the final Pareto archive is converted into a collection of binary perturbation masks. A pixel-wise density map is then computed by aggregating the occurrence frequency of perturbed locations across the archived solutions:

\[
D(i,j)=\frac{1}{|\mathcal{A}|}
\sum_{P\in\mathcal{A}} M_P(i,j),
\]

where \(\mathcal{A}\) denotes the final Pareto archive and \(M_P(i,j)\in\{0,1\}\) indicates whether pixel \((i,j)\) is perturbed in solution \(P\).

Finally, Gaussian smoothing is applied to obtain a spatially continuous density representation:

\[
D_{\mathrm{smooth}}=G_{\sigma}*D,
\]

where \(G_{\sigma}\) is a Gaussian kernel with standard deviation \(\sigma\), and \(*\) denotes convolution. The resulting density map is normalized and overlaid on the original image for visualization.

This map characterizes the spatial distribution of perturbation locations across the discovered Pareto solutions, rather than the perturbation pattern of a single adversarial example.

## Pipeline implementation

- Both GA and NSGA-II update `attacker.pareto_archive` after evaluating the initial
  population and every offspring population, before survivor selection. The
  archive is reset for each attack and is not limited to the final population or
  to successful adversarial examples.
- The archive minimizes the original `margin_loss` and `saliency_loss` values
  returned by the selected fitness function (including its existing sign
  conventions). It does not change fitness, selection, or the five attack return
  values. The existing front-score text output still describes the last
  population pool; the historical archive objectives are saved separately.
- Uniqueness means identical realized perturbations: changed pixel indices and
  their resulting RGB values. Genome ordering, zero-valued genes, and changes
  erased by clamping do not create extra solutions. Different RGB perturbations
  sharing the same binary mask remain distinct solutions. Equal objective values
  alone do not imply duplication.
- Each mask includes a pixel only if at least one channel actually differs from
  the clean image after clamping. A zero-perturbation solution contributes an
  empty mask but still counts in the archive denominator.
- `run_sample.py` and `run_batch.py` save the maps automatically. Pixel
  frequencies are averaged into a 4×4 grid, then bilinearly resized to the
  image size. Use `--density-sigma 1` (default, in pixels) for Gaussian
  smoothing, or `0` to disable smoothing. Smoothing uses reflective boundaries;
  normalization is linear min-max, with constant maps displayed as zero.
  No gamma or contrast exponent is applied. Existing saved visualizations must
  be regenerated to reflect linear normalization.
- The archive is uncapped to preserve the specified frequency distribution.
  Stored entries use sparse CPU snapshots, not full images or autograd graphs.
  Large Pareto fronts can still increase runtime and memory use.

### Saved outputs

For batch runs, each image folder contains:

- `perturbation_density_raw.png`: frequency-map heatmap (rescaled for display).
- `perturbation_density.png`: smoothed, normalized heatmap, using the same hot
  colormap as the existing saliency maps.
- `perturbation_density_overlay.png`: density overlay on the clean image.
- `perturbation_density.pt`: exact float tensors `grid_density`, `density`,
  `smoothed_density`, `normalized_density`, and `archive`. `grid_density` is
  the 4×4 per-cell mean changed-pixel frequency; the other maps are resized to
  image resolution. `normalized_density` is linear min-max normalized.
  Each archive entry contains flattened `pixels`, resulting
  RGB `values`, and the two `objectives`. Pixel indices are row-major,
  `row * width + column`.
- `perturbation_density.json`: archive size, image shape, sigma, objective/mask
  definitions, grid size, and output paths. The same metadata appears in
  `summary.json` under `perturbation_density` and in the batch report.

For single-image runs, these names are prefixed with the adversarial output stem:
`--output results/adv.png` produces `results/adv_perturbation_density.*` and
the corresponding `_raw.png` and `_overlay.png` files. Metadata is also returned
by `run_attack`.

Existing batch summaries retain the usual skip behavior. Use `--replace` to
regenerate old runs without density outputs, or to change the smoothing sigma.
These maps are class-independent archive statistics; there are no separate
class-A/class-B density maps. Insertion/deletion evaluation is not added to
these attack runners.

Focused regression tests: `python -m unittest discover -s tests -p test_perturbation_density.py`.