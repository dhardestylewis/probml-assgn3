# Image Descriptions and Specifications
**Created:** 2025-12-23
**Purpose:** Document expected content of all figures in STCS6701 Final Project

---

## Figure 1: Graphical Model (miwae_graphical_model.png)

**File:** `images/miwae_graphical_model.png`
**Referenced in Report:** Line 79, Figure~\ref{fig:graphical-model-miwae}
**Referenced in Poster:** Line 108

### Required Elements
- **Nodes:**
  - $c_i$: Discrete mixture indicator (categorical)
  - $z_i$: Latent variable (continuous, $K=3$ dimensions)
  - $x_i$: Observed covariates ($D=43$ dimensions)
  - $y_i$: Log-transformed sale price (scalar, conditionally observed)
- **Plates:**
  - Outer plate: $i = 1, \ldots, n$ (properties)
  - Inner plate (if applicable): Feature dimensions or mixture components
- **Edges:**
  - $c_i \rightarrow z_i$ (mixture component selects latent distribution)
  - $z_i \rightarrow x_i$ (latent generates covariates via decoder)
  - $z_i \rightarrow y_i$ (latent generates price via prediction network)
- **Parameters:**
  - $\pi$ (mixture weights) - global
  - $\theta$ (decoder, prediction network) - global

### Known Issues
- **TODO:** Line passing through text in graphical model needs to be fixed
- Suggested fix: Use graphviz with manual node positioning or increased `ranksep`

### Caption (Report Line 82)
"Graphical model of the semi-supervised MIWAE with a Student-t mixture prior."

---

## Figure 2: Convergence Diagnostics (miwae_loss_elbo.png)

**File:** `images/miwae_loss_elbo.png`
**Referenced in Report:** Line 117, Figure~\ref{fig:convergence-loss-miwae}
**Alternative Names:** `convergence_loss.png` (in poster images directory)

### Required Elements
- **X-axis:** Epoch (0 to 50)
- **Y-axis:** Negative ELBO (or Total Loss)
  - **TODO:** Be confident about labeling: "Total Loss" vs "-ELBO"
- **Lines:**
  - Training loss (solid line)
  - Validation loss (dashed or different color)
  - If multiple folds available, show individual fold lines (faint) + average (bold)
- **Observations:**
  - Loss stabilizes (plateaus) around epoch 50
  - Training and validation curves should converge

### Known Issues
- **TODO:** Confirm "Across Folds" vs "Single Fold" in title
- Current uncertainty: Are loss curves for one fold only or aggregated across folds?
- If single fold, remove "Across Folds" from title

### Caption (Report Line 117)
"Convergence diagnostics for the SemiSupMIWAE model: training and validation loss versus epoch."

---

## Figure 3: Residual Diagnostics (miwae_residuals_histogram.png, miwae_residuals_qq.png)

**Files:**
- `images/miwae_residuals_histogram.png`
- `images/miwae_residuals_qq.png`

**Referenced in Report:** Line 183, Figure~\ref{fig:residuals-log}

### Histogram (miwae_residuals_histogram.png)

#### Required Elements
- **X-axis:** Residual in log-price space ($y_{\text{true}} - y_{\text{pred}}$)
- **Y-axis:** Density or frequency
- **Overlays:**
  - **TODO:** Add Student-t density curves with varying $\nu$ (degrees of freedom)
  - **Goal:** Overlay curves with $\nu$ corresponding to kurtosis 9.27 and others for reference
  - **Rationale:** Visualize dataset kurtosis against theoretical heavy-tailed distributions
- **Annotations:**
  - Kurtosis = 9.27 (exact value, no approximation)
  - Optionally: Mean, median, standard deviation

#### Current Status
- **TODO:** Revise existing histogram to include Student-t density overlays
- **Blocked:** Requires Colab execution to regenerate

### QQ Plot (miwae_residuals_qq.png)

#### Required Elements
- **X-axis:** Theoretical Normal quantiles
- **Y-axis:** Sample residual quantiles
- **Reference Line:** 45-degree line (y = x)
- **Overlays:**
  - **TODO:** Add theoretical confidence bands (95% bootstrap or Kolmogorov-Smirnov bounds)
  - **Goal:** Enhance statistical rigor of QQ plot
- **Range:**
  - **TODO:** Confirm both axes span -10 to 10 range

#### Current Status
- **TODO:** Add confidence bands to QQ plot
- **Blocked:** Requires Colab execution

### Caption (Report Line 184)
"Residual diagnostics for $\log(\text{sale\_price})$: histogram and QQ-plot versus a Normal reference, illustrating heavy tails."

---

## Figure 4: Latent Space by Price (miwae_latents_by_sale_price.png)

**File:** `images/miwae_latents_by_sale_price.png`
**Referenced in Report:** Line 199, Figure~\ref{fig:latent-price}
**Alternative Names:** `latent_space_price.png` (poster)

### Required Elements
- **X-axis:** $z_3$ (1st principal latent, 49.9% variance)
- **Y-axis:** $z_1$ (2nd principal latent, 30.2% variance)
  - **TODO:** Fix latent labeling - change z0/z1 to z3/z1 by variance contribution
- **Color:** Sale price deciles (10 bins, sequential colormap)
- **Points:** Scatter plot of posterior means $\mu_z(x_i, y_i)$ for properties with observed $y_i$
- **Enhancements (Optional):**
  - **TODO:** Add density contours or convex hulls to improve readability
  - **TODO:** Generate supplementary plot colored by square footage to validate "Value/Size" interpretation
  - **Contingency:** If "Size" correlation not evident, DROP "Value/Size" claim from poster

### Known Issues
- **TODO:** Verify "Value/Size factor" claim
  - Report (Line 205): "single factor orders properties along a 'value/size' continuum"
  - Poster (Line 165): "SHAP attribution reveals primary latent encodes recent market trends and sales volume"
  - **Discrepancy:** Poster says trends/volume; report says value/size
  - **Action:** Confirm via SHAP analysis or drop ambiguous claim

### Caption (Report Line 201)
"Latent space visualization: posterior means $(z_0, z_1)$ colored by sale-price deciles. A smooth gradient along one axis indicates that one latent dimension captures a value/size continuum."

**TODO:** Update caption to reflect correct latent indices (z3, z1) and verified interpretation

---

## Figure 5: Latent Space by Building Class (miwae_latents_by_building_class.png)

**File:** `images/miwae_latents_by_building_class.png`
**Referenced in Report:** Line 211, Figure~\ref{fig:latent-building-class}
**Alternative Names:** `latent_space_bldg.png` (poster)

### Required Elements
- **X-axis:** $z_3$ (1st principal latent)
- **Y-axis:** $z_1$ (2nd principal latent)
- **Color:** Building class (categorical: 1-4 family, large elevator, mixed-use, etc.)
  - **TODO:** Decide whether to drop, organize, or aggregate building classes for clarity
- **Points:** Scatter plot of posterior means
- **Enhancements (Optional):**
  - Density contours per class
  - Convex hulls to show class boundaries

### Observations
- **Report (Line 207):** "Building classes form identifiable striations"
- **TODO:** Verify claim - user noted "I don't see clusters separating by building classes. I do see identifiable regions or striations"
  - **Action:** Update language from "clusters" to "striations" (already done in report)

### Caption (Report Line 212)
"Latent space visualization: posterior means $(z_0, z_1)$ colored by building class (e.g., 1--4 family, large elevator, mixed-use). Distinct clusters suggest that the latent factors encode property-type structure."

**TODO:**
- Update caption to use correct latent indices (z3, z1)
- Change "Distinct clusters" to "Identifiable striations" to match observations
- Change "latent factors" to "latent variables" for consistency

---

## Additional Figures (Poster Only)

### Spatial Residual Map (residuals_spatial_map.png)

**File:** `images/residuals_spatial_map.png`
**Referenced in Poster:** Line 148 (commented out)

#### Required Elements
- **Map:** NYC geographic boundaries
- **Points/Heatmap:** Residuals ($y_{\text{true}} - y_{\text{pred}}$) by location
- **Color:** Diverging colormap (e.g., blue = underpredict, red = overpredict)
- **Annotations:** Identify regions with concentrated geographic bias

#### Purpose
- **Poster (Line 145):** "Spatial plotting reveals bias"
- Demonstrates model errors are not uniformly distributed
- Policy relevance: Systematic undervaluation in specific neighborhoods

### Residuals vs References (residuals_hist_references.png)

**File:** `images/residuals_hist_references.png`
**Referenced in Poster:** Line 150 (commented out)

#### Required Elements
- **X-axis:** Residual in log-price space
- **Y-axis:** Density
- **Overlays:** Student-t density curves with varying $\nu$
  - Label curves: "Student-t $\nu = $ [value] (best fit, if applicable)"
  - Use Greek symbol for $\nu$
- **Footnote:**
  - $r = \log(y_{\text{true}}) - \log(y_{\text{pred}})$ (mathematical notation)
  - **TODO:** Ensure written in actual mathematical notation, not pseudo-code

### Coverage by Sale Year (coverage_by_sale_year.png)

**File:** `images/coverage_by_sale_year.png`
**Purpose:** Uncertainty calibration analysis

#### Required Elements
- **X-axis:** Sale year
- **Y-axis:** Empirical coverage (should span 0 to 1)
  - **TODO:** Fix y-axis range to 0-1
- **Reference Line:** Nominal coverage level (e.g., 0.95 for 95% prediction intervals)
- **Points/Line:** Actual coverage achieved by model per year

### QQ Plot (residuals_qq.png, residuals_qq_simple.png)

**Files:**
- `images/residuals_qq.png`
- `images/residuals_qq_simple.png`

#### Required Elements
- **Range:** -10 to 10 for both axes
  - **TODO:** Confirm current plots meet this requirement
- **Discussion Block (Poster):** Include QQ plot to show tail behavior

### Kurtosis Comparison (kurtosis_comparison.png)

**File:** `images/kurtosis_comparison.png`
**Purpose:** Graphical depiction of kurtosis

#### Required Elements
- **Curves:** Overlaid distributions with kurtosis 5, 9, 15 (or similar range)
- **Reference:** Normal distribution (kurtosis = 3)
- **Highlight:** Kurtosis = 9.27 (our data)

---

## Figures NOT Currently Included (Potential Additions)

### 1. Supervised Head Diagram
**TODO (from critiques.md):**
- "Aren't some diagrams required if we are including a Supervised Head?"
- Graphical model shows probabilistic half
- Standard deep learning diagram needed for supervised portion of MIWAE

### 2. Latent by Square Footage
**TODO:**
- Generate plot: Latents (z3, z1) colored by square footage (or similar size metric)
- Purpose: Confirm "Value/Size" interpretation
- If correlation not evident, drop claim

### 3. SHAP Feature Importance
**TODO (Poster mentions SHAP):**
- Poster (Line 165): "SHAP attribution reveals primary latent encodes recent market trends (26%) and building sales volume (19%)"
- No SHAP plot currently included
- Consider adding SHAP summary plot if space permits

---

## Consistency Checks

### Latent Variable Indexing
- **Report (Line 196):** "plot the top two dimensions by variance contribution, $(z_3, z_1)$"
- **Report Captions (Lines 201, 212):** Still say "$(z_0, z_1)$"
- **Poster (Line 162):** "$z_3$ (49.9\%) and $z_1$ (30.2\%)"
- **TODO:** Update all captions and text to use $(z_3, z_1)$ consistently

### Value/Size Factor
- **Report (Line 205):** "value/size continuum"
- **Poster (Line 165):** "recent market trends (26%) and building sales volume (19%)"
- **TODO:** Reconcile these interpretations
  - Either provide evidence for "value/size" (size plot)
  - Or update report to match poster ("trends/volume")

### Metrics Consistency
- **Poster (Line 134):** Avg. Log Posterior = -2.28
- **TODO (from prompt logs):** "Poster -1.15 vs Logs -2.2757"
- **Action:** Verify exact value from logs, ensure poster and report match

### Figure Numbering and Enumeration
- **TODO (critiques.md):** "Should both figures have the same caption? Should the figures captions be enumerated?"
- Current status: Figures are numbered (Figure 1, 2, etc.) in report
- Poster uses unnumbered captions
- **Action:** No change needed; standard practice differs between formats

---

## Plot Generation Scripts

### Known Scripts
1. `hws/hw3.d/generate_graph.py` - Generates graphical model
2. `hws/hw3.d/generate_kurtosis_plot.py` - Generates kurtosis comparison
3. `hws/hw3.d/create_logos.py` - Creates Columbia logos

### Missing Scripts
- **TODO:** Identify Python script that produces each flagged figure
- Likely source: `code/post_training_eval.py` or similar evaluation script
- **Action:** Review codebase or user-provided notebook to locate

### Regeneration Requirements
- **Blocked:** User stated "you will not be able to regenerate any of the existing plots on this machine. that requires more back and forth with me to update code on a separate colab instance."
- **TODOs requiring regeneration:**
  - Student-t overlays on histogram
  - Confidence bands on QQ plot
  - Latent plots with corrected indices
  - Size-colored latent plot
  - Any other plot revisions

---

## Summary of Critical TODOs

### High Priority (P1)
1. **Fix latent indexing:** z0/z1 → z3/z1 throughout report and poster
2. **Verify metrics:** -1.15 vs -2.2757 log posterior
3. **Reconcile interpretations:** Value/Size vs Trends/Volume for latent axis

### Important (P2)
4. **Regenerate histogram:** Add Student-t density overlays (kurtosis 9.27)
5. **Regenerate QQ plot:** Add confidence bands, verify -10 to 10 range
6. **Fix graphical model:** Remove line passing through text
7. **Clarify loss plot:** "Across Folds" vs "Single Fold"

### Polish (P3)
8. **Add supervised head diagram:** If required for completeness
9. **Generate size-colored latent plot:** To validate or drop "Size" claim
10. **Update all captions:** Ensure consistency with figure content and indexing

---

## Document Metadata
- **Last Updated:** 2025-12-23
- **Status:** Most figures exist but require revisions per user feedback
- **Regeneration:** Blocked on Colab access; coordinate with user for updates
