# HW3 Project TODO
**Target Deadline: 4 hours from 2025-12-23 16:12 (~20:12)**

---

## P1 - Critical: Latent Plot Subset Identity (Goal 1)

- [ ] **Priority:** Create detailed pipeline diagrams for the model architecture (Future)

### P1.1 - Base Mask Definition [Added: 2025-12-23 16:12]
- [ ] Make one authoritative `base_eval_mask` over `df_pred` rows that is the only admissible starting point for evaluation plots [Added: 2025-12-23 16:12]
- [ ] `base_eval_mask`: boolean length `len(df_pred)`, True exactly at `eval_pos_idx` (CV held-out or fallback hold-out) [Added: 2025-12-23 16:12]
- [ ] `mask_has_price`: finite `log_y_col` and within the global filter bounds [Added: 2025-12-23 16:12]
- [ ] `mask_has_bldg`: building class `notna` if needed [Added: 2025-12-23 16:12]
- [ ] `mask_plot_price = base_eval_mask & mask_has_price & mask_has_bldg` (or drop `mask_has_bldg` if price plots do not require it, but then document it) [Added: 2025-12-23 16:12]
- [ ] `mask_plot_class = base_eval_mask & mask_has_bldg` (price not required unless you want identical points) [Added: 2025-12-23 16:12]
- [ ] Define `mask_has_price_eval` and `mask_has_bldg_eval` on `df_eval`, not `df_pred`, and define `mask_plot_price_eval` and `mask_plot_class_eval` in eval-index space only [Added: 2025-12-23 16:44]
- **Invariant**: No plot receives a `df_pred`-space mask.
- **Assertion location**: All `plot_*` functions; **Failure message**: `"Plot mask length {len(mask)} != df_eval length {len(df_eval)}; received df_pred-space mask"`

### P1.2 - Remove Duplicate Mask Blocks / Hidden-State Dependencies [Added: 2025-12-23 16:12]
- [ ] Compute `mu_z` first [Added: 2025-12-23 16:12]
- [ ] Build masks next, using `base_eval_mask` [Added: 2025-12-23 16:12]
- [ ] Then generate all latent plots [Added: 2025-12-23 16:12]
- [ ] Remove the earlier "Mask debug" block that references `mu_z` before it is defined, or move it after `mu_z` and make it compare different masks (price vs class) rather than comparing a mask to itself [Added: 2025-12-23 16:12]

### P1.3 - Plot Contract Banner Function [Added: 2025-12-23 16:12]
- [ ] Add a plot contract banner function and call it on every figure [Added: 2025-12-23 16:12]
- [ ] Every figure must self-report: mask name, `n_total_eval`, `n_used`, and the main drop reasons (missing z, missing price, failed global price filter, missing building class if required) [Added: 2025-12-23 16:12]
- [ ] The plot banner should read counts from `eval_audit_df`, not recompute them ad hoc [Added: 2025-12-23 16:21]

### P1.3a - Hard-Stop Defect F: Global Filter Call Correctness [Added: 2025-12-23 16:21]
- [ ] Apply global price filter only to the eval subset (`df_eval`), never to `df_pred` [Added: 2025-12-23 16:21]
- [ ] `apply_global_price_filter` asserts input length equals eval subset length; call sites must pass `df_eval` and eval-length arrays only [Added: 2025-12-23 16:21]
- **Invariant**: After `df_eval` construction, no function receives `df_pred`-length arrays.

### P1.3b - Hard-Stop Defect D: PIT/Coverage Control-Flow [Added: 2025-12-23 16:21]
- [ ] Move all PIT/coverage computations into a single unconditional post-filter block that always executes once eval data are constructed [Added: 2025-12-23 16:21]
- [ ] No PIT/coverage logic inside residual plot blocks or any optional plot branch [Added: 2025-12-23 16:21]
- **Invariant**: PIT/coverage runs exactly once, unconditionally, after eval subset is finalized.

### P1.3c - Hard-Stop Defect E: QQ/Helper Redefinition Runtime Defects [Added: 2025-12-23 16:21]
- [ ] Remove the second definition of `digitize_safe`; define once [Added: 2025-12-23 16:21]
- [ ] Fix `normal_ppf` usage so it is applied elementwise (or replace with a vector-safe implementation) [Added: 2025-12-23 16:21]
- [ ] Add a smoke test that QQ plot code runs end-to-end [Added: 2025-12-23 16:21]
- [ ] Add `eval_smoke_test()`: constructs `df_eval`, builds `eval_audit_df`, runs PIT/coverage block, generates at least one latent plot and one residual plot, and asserts: [Added: 2025-12-23 16:44]
  - `len(df_eval) == sum(base_eval_mask)`
  - All derived arrays have length `len(df_eval)`
  - All plot masks have length `len(df_eval)`
  - Plot banner counts match `eval_audit_df` counts
- **Invariant**: No helper function is defined more than once; all helpers are vector-safe.

### P1.3d - Eval-Subset-Only Dataflow Contract [Added: 2025-12-23 16:21]
- [ ] Construct `df_eval = df_pred[base_eval_mask].copy()` exactly once [Added: 2025-12-23 16:21]
- [ ] From that point onward, every derived array must be eval-length (`z`, `mu_log_eval`, `y_log_eval`, `residuals`, etc.) [Added: 2025-12-23 16:21]
- [ ] Every plotting function accepts `df_eval` plus eval-length arrays, and optional plot masks are defined in eval-index space (length `len(df_eval)`), not `df_pred` space [Added: 2025-12-23 16:21]
- [ ] Add assertion: `assert len(array) == len(df_eval)` for all derived arrays after `df_eval` construction [Added: 2025-12-23 16:21]
- **Invariant**: All arrays after `df_eval` construction have length `len(df_eval)`.

### P1.3e - Row-Level Audit Artifact [Added: 2025-12-23 16:21]
- [ ] Create `eval_audit_df` with one row per `df_eval` row and boolean columns: `has_z`, `has_price`, `passes_global_filter`, `has_bldg`, `in_plot_price`, `in_plot_class`, plus a categorical `drop_reason` (first-failure or multi-label) [Added: 2025-12-23 16:21]
- [ ] Save `eval_audit_df` (CSV or Parquet) alongside plots [Added: 2025-12-23 16:21]
- **Invariant**: `eval_audit_df` is the single source of truth for all plot contract banners.

---

## P1 - Critical: Plot Suite Completion (Goals 2 & 3)

### P1.4 - Missing Plots [Added: 2025-12-23 16:12]
- [ ] **P0**: No single audit artifact that all plots reference. No per-plot contract banner with `n_used` and mask name [Added: 2025-12-23 16:12]
- [ ] **P1**: No latent-space overlay of "excluded points" vs "included points" for each plot mask [Added: 2025-12-23 16:12]
- [ ] **P6**: No latent heatmap of residual median and residual scale [Added: 2025-12-23 16:12]
- [ ] **P7**: No coverage/PIT by z-region (4x4 quantile bins, etc.) [Added: 2025-12-23 16:12]
- [ ] **P9**: No multi-seed or multi-fold stability panels with fixed limits and binning [Added: 2025-12-23 16:12]

### P1.5 - Partial Plots Needing Fixes [Added: 2025-12-23 16:12]
- [ ] **P2** (latent density hexbin): Uses `mask_common` (`df_pred`-wide), not eval subset. Axis limits not guaranteed identical across every latent plot call site [Added: 2025-12-23 16:12]
- [ ] **P3** (price scatter + hexbin): The hexbin uses mean log-price, but no explicit `mincnt` thresholding strategy tied to a coverage plot [Added: 2025-12-23 16:12]
- [ ] **P4** (density-only hexbin): Exists, but not paired as a companion with identical binning and displayed adjacent to P3 in the same figure or at least in a linked pair with identical parameters [Added: 2025-12-23 16:12]
- [ ] **P5** (building class plot): Historically had point-count mismatch; current code does not show a definitive, single-source mask definition for class vs price plots [Added: 2025-12-23 16:12]
- [ ] **P8** (variance-based dim selection): Variance is not "importance for y." Perm importance helpers exist but not used to justify plotted dims or to report uncertainty [Added: 2025-12-23 16:12]

### P1.6 - Dimension Selection Improvement [Added: 2025-12-23 16:12]
- [ ] Stop selecting plotted latent dimensions by variance if the goal is "structure that matters for price and error" [Added: 2025-12-23 16:12]
- [ ] Choose dims that maximize correlation with `mu_log_eval` (or with residual magnitude) on the eval subset, and report those correlations in the plot contract [Added: 2025-12-23 16:12]
- [ ] Then replace that with true y-head importance once the head-vs-trainer gate is stable [Added: 2025-12-23 16:12]

### P1.7 - Paired Value+Counts Binning [Added: 2025-12-23 16:12]
- [ ] Make coverage first-class by pairing value with counts using identical binning [Added: 2025-12-23 16:12]
- [ ] **P3**: median log(price) per bin, blank bins with count < k [Added: 2025-12-23 16:12]
- [ ] **P4**: log10(count) per bin, same bins, same axis limits, same gridsize [Added: 2025-12-23 16:12]

---

## P1 - Critical: Poster and Report Revisions

### P1.8 - Layout and Visual Fixes [Added: 2025-12-23 15:40]
- [ ] Reposition QQ plot to top-right in 2x2 grid (spatial left, residuals vs refs bottom-right) [Added: 2025-12-23 15:40]
- [ ] Fix latent labeling: change z0/z1 to z3/z1 (by variance 49.9%, 30.2%) [Added: 2025-12-23 15:40]
- [ ] Enlarge Columbia logos and add clickable links [Added: 2025-12-23 15:40]
- [ ] Link STCS 6701 to course webpage, Prof. Blei to profile [Added: 2025-12-23 15:40]

### P1.9 - Content Accuracy Fixes [Added: 2025-12-23 15:40]
- [ ] Remove/revise "Value/Size factor" - SHAP: Recent Trend 26%, Bldg Sales Vol 19% [Added: 2025-12-23 15:40]
- [ ] Change "clusters separate" to "identifiable striations by building class" [Added: 2025-12-23 15:40]
- [ ] Verify metrics: Poster -1.15 vs Logs -2.2757 [Added: 2025-12-23 15:40]
- [ ] Copy hw3-SUBMITTED.tex to report.tex with corrected metrics [Added: 2025-12-23 15:40]

### P1.10 - Plot Generation Updates [Added: 2025-12-21 21:40]
- [ ] "get rid of this 'Visualizing Residuals' plot and instaed begin planning in a single more involved outlined TODO how to revise the existing plot to include some of these same reference Student T kurosis lines for reference against ourr own dataset"
    -   **Detailed Plan**: Revise `miwae_residuals_histogram.png` generation code:
        -   Goal: Directly visualize dataset kurtosis against theoretical heavy-tailed distributions.
        -   Action: Overlay Student-t density curves with varying degrees of freedom (e.g., $\nu$ corresponding to Kurtosis ≈ 9.27 and others for reference) on top of the residual histogram/KDE.
        -   Rationale: Replaces the standalone normal/heavy-tail comparison with a data-integrated view. [Added: 2025-12-21 21:40]
- [ ] "likewise all the other plots after reviewing them visually inspecting them"
    -   **Detailed Plan**: Revise `miwae_residuals_qq.png`:
        -   Goal: Enhance statistical rigor of the Q-Q plot.
        -   Action: Add theoretical confidence bands (e.g., 95% bootstrap or Kolmogorov-Smirnov bounds).
    -   **Detailed Plan**: Revise `miwae_loss_elbo.png`:
        -   Goal: Verify and clarify "Across Folds" vs "Single Fold".
        -   Action: Plot faint lines for individual folds and a bold average line (if available), or clarify labeling.
    -   **Detailed Plan**: Revise Latent Space Plots (`miwae_latents_*.png`):
        -   Goal: Improve readability and validate "Value/Size" interpretation.
        -   Action: Add density contours/convex hulls. **Crucially**, generate a new plot "Latents colored by Square Footage" (or similar size metric) to confirm the "Size" factor claim.
        -   Contingency: If "Size" correlation is not visually evident or space is limited, **DROP** the "Value/Size" text claim from the poster.
    -   **Detailed Plan**: Revise `miwae_graphical_model.png`:
        -   Goal: Fix visual artifact (line passing through text).
        -   Action: Use `graphviz` with manual node positioning or increased `ranksep`. [Added: 2025-12-21 21:40]

### P1.11 - Poster Verification [Added: 2025-12-23 16:12]
- [ ] Run code/post_training_eval.py to generate residuals_standardized_qq.png (using updated residuals_standardized_qq logic) [Added: 2025-12-23 16:12]
- [ ] Verify exact Kurtosis value (was 9.27?) and R^2 (was 0.27?) [Added: 2025-12-23 16:12]
- [ ] Update hws/hw3.d/poster.tex with:
    - New residuals_standardized_qq.png (replaces residuals_qq_simple.png)
    - Verified numbers for Kurtosis and R^2 [Added: 2025-12-23 16:12]

---

## P1 - Critical: References and Notation

### P1.12 - Citations Required [Added: 2025-12-21 21:40]
- [x] "doesnt this require a reference? the Missing-data Importance-Weighted Autoencoder (MIWAE)" [Completed: 2025-12-23 21:15]
- [x] "likewise Student-t Mixture" [Completed: 2025-12-23 21:15]

### P1.13 - Symbol/Notation Verification [Added: 2025-12-21 21:40]
- [ ] "would the community automatically understand diag? search to be certain" [Added: 2025-12-21 21:40]
- [ ] "have you provided a legend in fine print or footnote of all these symbols if necessary or at least of any symbols beyond the usual competence of this field using search to establish that?" [Added: 2025-12-21 21:40]
- [ ] "is invoking a function like Mixture-of-Student-T sufficient?" [Added: 2025-12-21 21:40]

### P1.14 - Metric Presentation Verification [Added: 2025-12-21 21:40]
- [ ] "Avg. Log Posterior p(y |x) −1.15" - still not clear to me this is the usual way confirmed by search, downloding relevant reference publications to a dedicated folder, converting to txt, inspecting those, that this is to convery this specific information [Added: 2025-12-21 21:40]
- [ ] "is this usually how this value is presented? search up comparable papers which do, store them in a reference publications folder, convert to txt, check both via vision and txt and let me know" [Added: 2025-12-21 21:40]

---

## P1 - Critical: Diagram and Figure Requirements

### P1.15 - Graphical Model and Supervised Head [Added: 2025-12-21 21:40]
- [ ] "arent some diagrams required if we are including a Supervised Head? graphical model for the probabilistic half and standard deep learning graphs/diagrams for the supervised portion of MIWAE" [Added: 2025-12-21 21:40]
- [ ] "any way to fix the line passing through text in graphical model?" [Added: 2025-12-21 21:40]

### P1.16 - Caption and Labeling Consistency [Added: 2025-12-21 21:40]
- [ ] "should both figures have the same caption? shoudl the figures captions be enumerated?" [Added: 2025-12-21 21:40]
- [ ] "is log-price space important enough to include in the title or axes titles of any plots rather than as part of the caption or a footnote?" [Added: 2025-12-21 21:40]
- [ ] "should we anywhere ever mention the `log_y_true`, `log_y_pred` variables we used internally? or translate them or instead include either in the methodology or in the caption alongside the figure or both?" [Added: 2025-12-21 21:40]
- [ ] "do we need any more specific labels than \"z1\", \"z2\"?" [Added: 2025-12-21 21:40]
- [ ] "that latents referred to in the caption are different Figure: Latents (z0, z1) colored by Sale Price Decile. isnt this information duplicative? is there anything more specific we should be pointing out?" [Added: 2025-12-21 21:40]
- [ ] "isn't \"colored by sale-price deciles\" repetitive of what the plot is actually doing?" [Added: 2025-12-21 21:40]
- [ ] "need include \"truncated\" in title?" [Added: 2025-12-21 21:40]

### P1.17 - Kurtosis Visualization [Added: 2025-12-21 21:40]
- [ ] "is tehre a graphical way to depict the kurtosis? i am not visually tuned in enough to know what a kurtosis of 9 vs 5 vs 15 might look like? should we be included any such curves for reference?" [Added: 2025-12-21 21:40]
- [ ] "eal estate data is heavy-tailed  (kurtosis ≈ 9.27)" [Added: 2025-12-21 21:40]

---

## P1 - Critical: Methodology and Claims

### P1.18 - Prior/Posterior Confirmation [Added: 2025-12-21 21:40]
- [ ] "confirm the prior and posterior we used" (Re-verify with User) [Added: 2025-12-21 21:40]
    -   Note: "we will need to confirm the prior and posterior that is a high priority TODO"
- [ ] "Gaussian posterior when i provide the nb and code" [Added: 2025-12-21 21:40]

### P1.19 - Evaluation Setup Details [Added: 2025-12-21 21:40]
- [ ] "shouldnt you be more specific about the set up describing precisely the whole thing? on a 20% held-out set." [Added: 2025-12-21 21:40]
- [ ] "have we confirmed 'on a Representative Fold'? is this caption repetitive of information provided elsewhere? does it matter? push back" [Added: 2025-12-21 21:40]

### P1.20 - Claims Requiring Evidence [Added: 2025-12-21 21:40]
- [ ] "look at the plot visually, do you agree with your interpretation that \"smooth gradient suggests one axis encodes a "Value/Size" continuum.\"" [Added: 2025-12-21 21:40]
- [ ] "or \"captures interpretable structure\"" [Added: 2025-12-21 21:40]
- [ ] "is there enough information from the plots here to reach this interpretation? \"suggests a "Value/Size" factor\"? will we have enough information by slightly modifying existing plots or enough space to include any further plots to claim this? if not should we drop this claim?" [Added: 2025-12-21 21:40]
- [ ] "where is this gradient in the existing plot visually? if nowhere extend the existing TODO to reflect updates to the plot we will be making" [Added: 2025-12-21 21:40]
- [ ] "Gradient z0/z1" [Added: 2025-12-21 21:40]

### P1.21 - Building Class and Price Display [Added: 2025-12-21 21:40]
- [ ] "should we drop the building classes or organize them or aggregate them?" [Added: 2025-12-21 21:40]
- [ ] "should we have converted out the sale price deciles to actual values written out in $?" [Added: 2025-12-21 21:40]

---

## P1 - Critical: Equations and Text

### P1.22 - Equations Selection [Added: 2025-12-21 21:40]
- [ ] "the equations we have selected to inlcude/exclude relative to the HW" [Added: 2025-12-21 21:40]

### P1.23 - Text Style Fixes [Added: 2025-12-21 21:40]
- [x] "you are still repeating log- everywhere" [Completed: 2025-12-23 21:10]
- [ ] "be confident about either titling the axis generally \"total loss\" or specifically-ELBO but not both... make reasoning to decide which" - in the TODOs [Added: 2025-12-21 21:40]
- [x] "never use word approximately, deliberately confidently abbreviate the number instead 'approximately 122,000 NYC'" [Completed: 2025-12-23 20:45]
- [ ] "'Loss stabilizes around 50 epochs' - isnt this for the caption? its stable throughout, more like plateaus" [Added: 2025-12-21 21:40]
- [x] "Predicts log-transformed sale prices" [Completed: 2025-12-23 21:10]
- [x] "you can just say predicts sale prices and mention somewhere where directly required for interpretation that everything is log-transformed" [Completed: 2025-12-23 21:10]

---

## P1 - Critical: Blocked Items (Require User/Colab)

### P1.24 - Blocked on External Access [Added: 2025-12-21 21:40]
- [ ] "examine my images we are including, what needs to be updated, how it should be revised," [BLOCKED: Requires User/Colab] [Added: 2025-12-21 21:40]
- [ ] "our loss curves are for fold only? does that matter? especially since the title refers to \"across folds\"?" [BLOCKED: Requires User/Colab] [Added: 2025-12-21 21:40]

---

## P1 - Critical: Documentation and Meta

### P1.25 - Logs and Guidelines [Added: 2025-12-21 21:40]
- [ ] "ensure all of this is reflected in your prompts-logs and guidelines todo and suggest any other md we might need if any as appropriate for when we go to write the final report" [Added: 2025-12-21 21:40]
- [ ] "\"back-transformed to $\" is an unnecessary detail are you learning anything of the way you have been writing recording that and including that in a meta you refer to each and every prompt to improve your writing in the future?" [Added: 2025-12-21 21:40]
- [ ] "\", or SemiSupMIWAE,\" this is one exception i permitted use of paranthesis did you record that anywhere for your future writing?" [Added: 2025-12-21 21:40]
- [ ] ", or SemiSupMIWAE," [Added: 2025-12-21 21:40]
- [ ] "have you been reading your METAs each and every time? i see limited evidence of such in your chat responses" [Added: 2025-12-21 21:40]
- [ ] Review new GUIDELINES.md Section 0 (Version Control Safety) [Added: 2025-12-11 23:16]

### P1.26 - User Communication Items [Added: 2025-12-21 21:40]
- [ ] "you didnt respond directly to me in chat regarding my core questions surrounding equitable taxation (fair revenue generation) and urban planning (optimizing physical/service infrastructure)." [Added: 2025-12-21 21:40]
- [ ] "we havent sufficiently conversed about - is \"urban planning\" the main other field beyond equitable taxation that we are targetting... one seems to be a field and the other an action" [Added: 2025-12-21 21:40]
- [ ] "one is a broader claim that demands citation, the other is specifically measured from our dataset, be clear about both" [Added: 2025-12-21 21:40]

---

## P1 - Critical: Branding and Layout [Added: 2025-12-23 15:40]

### P1.27 - Logo and Formatting [Added: 2025-12-21 21:40]
- [ ] "wheres the overall title... logos... date... class name... professor's name?" - do you no longer have access to prev prompts in this conversation to write out unabbreviated? [Added: 2025-12-21 21:40]
- [ ] "did you pull and use the Columbia logo and letterhead from where they provide such materials on their website? did you use the actual name for the course as found by searching Blei's 2025F? date it december 23 2025" [Added: 2025-12-21 21:40]
- [ ] "greater vertical spread across poster" [Added: 2025-12-21 21:40]

---

## P2 - Important Refinements

### P2.1 - Meta-Process Review [Added: 2025-12-23 16:25]
- [ ] Review GUIDELINES PD.6 (Conversation Improvement Meta-Reflection) and mark as (REVIEWED) [Added: 2025-12-23 16:25]

### P2.2 - Reference Research and ICML Formatting [Added: 2025-12-23 16:30]
- [ ] Download similar MIWAE/VAE for real estate papers to `final_project/references/` [Added: 2025-12-23 16:30]
- [x] Convert report to ICML 2025 style (icml2025.sty, two-column, 10pt Times, APA references) [Completed: 2025-12-23 21:00]
- [x] Add Impact Statement section (required for ICML) [Completed: 2025-12-23 21:00]
- [x] Ensure references are complete with page numbers [Completed: 2025-12-23 21:15]
- [x] Create `final_project/references/notes.md` with section-by-section recommendations for content, length, references [Completed: 2025-12-23 20:30]
- [ ] Identify appropriate venues: ICML, NeurIPS, AISTATS, UAI workshops [Added: 2025-12-23 16:30]

### P2.3 - Report Text Revisions (from critiques.md) [Added: 2025-12-23 16:41]
> See `final_project/critiques.md` for detailed item-level tracking.

**Summary of outstanding items (~65 total):**
- **Abstract (P2)**: Clarify panel size, noise assumptions, MIWAE terminology, discriminative baseline, citation specifics
- **Intro (P2)**: Blei vocabulary consistency, evidence for skewness claims
- **Model (P2)**: Contextualize Cholesky/logits, explain $K=3$ choice
- **Inference (P3)**: Define amortized VI, importance samples, notation consistency
- **Data/Setup (P3)**: Justify 80/20 split, add hyperparams (batch size, lr, seed)
- **Results (P3)**: Exact values vs approx, paragraph unity, claim quantification
- **Plans (P3)**: Remove meta-commentary, specify architectures, rename sections

### P2.4 - Image Review and Recompilation [Added: 2025-12-23 16:47]
> See `references/IMAGE_DESCRIPTIONS.md` for detailed visual inspection analysis.

- [ ] Review all 28 images against IMAGE_DESCRIPTIONS.md recommendations [Added: 2025-12-23 16:47]
- [ ] Implement critical improvements identified (add sample sizes, error bars, statistical tests) [Added: 2025-12-23 16:47]
- [ ] Regenerate plots from `code/post_training_eval.py` with improvements [Added: 2025-12-23 16:47]
- [ ] Update `final_project/images/` with improved versions [Added: 2025-12-23 16:47]
- [ ] Recompile `STCS6701_FinalProject_Report_Lewis.tex` and `STCS6701_FinalProject_Poster_Lewis.tex` [Added: 2025-12-23 16:47]
- [ ] Verify 15 publication-ready figures, exclude 4 draft figures identified [Added: 2025-12-23 16:47]

---

## P3 - Plans for Final Weeks (Roadmap to Implementation) [Added: 2025-12-23 17:33]

### P3.1 - Student-t Likelihood Implementation (Robustness)
- [ ] **Define Student-t Head**:
    - [ ] In `SemiSupMIWAE` (line ~4902), modify `__init__` to accept `likelihood_type="student_t"` and `df=4.0` (or learnable `df`).
    - [ ] Add `self.log_df = nn.Parameter(...)` if learnable degrees of freedom are desired (optional).
- [ ] **Implement Log-Likelihood Logic**:
    - [ ] In `VAETrainer._calculate_y_loss` (line ~5090), add `if self.price_loss_type_y == "student_t_nll":`.
    - [ ] Implement robust NLL: `log_gamma(...) - log_gamma(...) - 0.5*log(pi*nu*sigma^2) - (nu+1)/2 * log(1 + (resid^2)/(nu*sigma^2))`.
- [ ] **Update Configuration**:
    - [ ] Add `price_loss_type` argument to `SemiSupMIWAE` and `VAETrainer` calls in the pipeline script (`retabularautovae_probml.py`).

### P3.2 - Richer Price Head Architecture (Expressivity)
- [ ] **Expand `price_mean_head`**:
    - [ ] Currently `nn.Linear(latent_dim, y_dim)`.
    - [ ] Replace with `MLP(in=latent, hidden=[64, 64], out=y_dim, act=LeakyReLU)` in `SemiSupMIWAE.__init__`.
    - [ ] Add residual connections or BatchNorm if training instability occurs.
- [ ] **Enrich `price_var_head`**:
    - [ ] Currently `nn.Linear(latent_dim, y_dim)`.
    - [ ] Similar upgrade to MLP to allow uncertainty `sigma(z)` to depend non-linearly on latent structure.

### P3.3 - Rigorous Uncertainty Calibration
- [ ] **Metric Expansion**:
    - [ ] Compute "Expected Calibration Error" (ECE) for regression in `post_training_eval.py`.
    - [ ] Add `std_calibration_plot(y_true, y_mean, y_std)`: Bin by predicted uncertainty, compare RMS error vs Mean Predicted Uncertainty.
- [ ] **Validation**:
    - [ ] Compare `Gaussian` vs `Student-t` calibration curves.
    - [ ] Verify if "Heavier Tails" (P3.1) actually improve empirical coverage (Figure 5).

### P3.4 - Robust Metrics Computation
- [ ] **Implement Median Absolute Error (MedAE)**:
    - [ ] Already in `_report_metrics`; ensure it is logged to central CSV tracker.
- [ ] **Implement Tail Quantile Loss**:
    - [ ] Compute `QuantileLoss(q=0.95)` and `QuantileLoss(q=0.99)` to quantify worst-case performance.
    - [ ] Add to `eval_utils_snippet.py` or `post_training_eval.py` report block.

---

## Completed Tasks
*(See TODO-COMPLETED.md for full history in CHANGELOG order)*


- [x] Verify `HistGradientBoostingRegressor` implementation in code (Prompt 205) [Completed: 2025-12-23 20:20]
- [x] Verify Uncertainty Calibration implementation in code (Prompt 205) [Completed: 2025-12-23 20:20]
- [x] Audit PROMPTS-LOG.md for missing specific recent prompts (Prompt 206) [Completed: 2025-12-23 20:22]
