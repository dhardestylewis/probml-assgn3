# Comprehensive Image Descriptions for NYC Property Valuation Project

This document provides detailed descriptions of all figures in the Semi-Supervised MIWAE project for NYC property valuation. These descriptions are written to be sufficiently detailed to recreate the report and poster from text alone, meeting the standards of top-tier ML conferences (ICML, NeurIPS, AAAI).

**Author:** Generated via visual inspection on 2025-12-23
**Project:** Semi-Supervised Latent Variable Model for NYC Property Valuation
**Total Images:** 28

---

## Table of Contents

1. [Model Architecture](#1-model-architecture)
2. [Training Diagnostics](#2-training-diagnostics)
3. [Latent Space Visualizations](#3-latent-space-visualizations)
4. [Residual Analysis](#4-residual-analysis)
5. [Model Diagnostics](#5-model-diagnostics)
6. [Branding Assets](#6-branding-assets)

---

## 1. Model Architecture

### 1.1 miwae_graphical_model.png

**Type:** Probabilistic graphical model (plate notation)

**Description:**
A directed acyclic graph representing the generative process for a single property in the Semi-Supervised MIWAE model. The diagram uses standard plate notation with:

- **Outer plate**: Labeled "parcels (i = 1,...,n)" with dashed gray border, indicating replication across n properties
- **Inner plate**: Labeled "mixture components (k = 1,...,K)" with dashed gray border, containing mixture-specific parameters

**Nodes (from left to right):**
1. **π** (pi): Global mixture weights parameter (unshaded oval)
2. **c_i**: Mixture component indicator for property i (unshaded oval)
3. **μ_k, Σ_k**: Component-specific mean and covariance parameters for the Student-t mixture prior (unshaded ovals in mixture plate)
4. **z_i**: Latent variable (3-dimensional) for property i (unshaded oval)
5. **x_i**: Observed covariates (D=43 features) for property i (shaded oval with double border indicating observed variable)
6. **y_i**: Log-transformed sale price for property i (shaded oval with double border indicating observed variable)

**Edges (dependencies):**
- π → c_i: Mixture weights determine component assignment
- c_i → z_i: Component assignment determines which mixture component generates z_i
- μ_k, Σ_k → z_i: Component parameters influence latent variable generation
- z_i → x_i: Latent variables generate observed covariates through decoder
- z_i → y_i: Latent variables generate sale price through supervised head

**Key Technical Details:**
- The model uses a Student-t mixture prior (not explicitly shown but mentioned in surrounding text)
- Missing prices are handled by omitting the y_i → observation edge for properties without sales
- The double circle on x_i and y_i indicates observed variables
- Single circles indicate latent/parameter variables

**Assessment vs. Conference Standards:**
This figure meets conference standards with clear plate notation and proper graphical model conventions. However, it could be enhanced by:
- Adding notation for the number of mixture components (K=5 in implementation)
- Explicitly showing the missing data mechanism
- Including decoder/encoder network parameterization annotations

---

## 2. Training Diagnostics

### 2.1 miwae_loss_elbo.png

**Type:** Line plot showing convergence diagnostics

**Dimensions:** Approximately 6" × 4" (standard single-column width)

**Axes:**
- **X-axis:** "Epoch" ranging from 0 to 100 (linear scale)
- **Y-axis:** "Total loss (= -ELBO)" ranging from 0 to ~450 (linear scale)

**Data Series:**
1. **Training loss** (solid blue line): Shows rapid initial descent from ~450 at epoch 0, dropping to ~100 by epoch 10, then gradually converging to ~20 by epoch 50, remaining stable through epoch 100
2. **Validation loss** (dashed orange line): Tracks training loss closely with slightly higher values, starts at ~425, drops to ~90 by epoch 10, converges to ~25 by epoch 50

**Title:** "SemiSupMIWAE convergence: train vs val loss across folds"

**Legend:** Positioned in upper-right quadrant
- "train (fold 1)" - blue solid line
- "val (fold 1)" - orange dashed line

**Visual Pattern:**
- Characteristic VAE convergence curve with steep initial drop
- No evidence of overfitting (train and validation curves remain close)
- Convergence achieved by approximately epoch 40-50
- Minor fluctuations after convergence suggest stable optimization
- Y-axis shows negative ELBO (i.e., loss to minimize)

**Grid:** Light gray gridlines at major tick intervals on both axes

**Annotation at bottom:** "Total Loss = Reconstruction Loss + beta * KL Divergence"

**Key Insights:**
- Model converges in ~50 epochs as stated in report
- Close tracking of train/validation suggests good generalization
- Stable plateau indicates successful optimization without collapse

**Assessment vs. Conference Standards:**
Good diagnostic plot. To meet highest standards, should add:
- Error bands/shading showing variance across multiple runs or folds
- Final converged loss values annotated
- Learning rate schedule if used
- Batch size and other training hyperparameters in caption

---

### 2.2 convergence_loss.png

**Type:** Line plot showing training convergence

**Dimensions:** Similar to miwae_loss_elbo.png but with cleaner styling

**Axes:**
- **X-axis:** "Epoch" ranging from 0 to 100 (linear scale)
- **Y-axis:** "Total Loss" ranging from 0 to ~500 (linear scale)

**Data Series:**
1. **Train** (solid blue line with markers): Starts at ~475, drops steeply to ~100 by epoch 15, converges to ~20 by epoch 40
2. **Validation** (dashed orange line with square markers): Starts at ~350, follows similar trajectory, converges to ~25 by epoch 40

**Title:** "SemiSupMIWAE Convergence"

**Legend:** Positioned in upper-right
- "Train" - blue solid
- "Validation" - orange dashed

**Visual Pattern:**
- Similar to miwae_loss_elbo.png but with slightly different initial values
- Both curves show smooth exponential-like decay
- Validation starts lower than training (unusual, may indicate different fold or subsample)
- Convergence well-established by epoch 40
- Training and validation converge to nearly identical values (~20-25)

**Grid:** Clean white background with light gray gridlines

**Differences from miwae_loss_elbo.png:**
- Cleaner typography and styling (likely created later)
- Different initial loss values suggest different fold or random seed
- Better visual clarity for presentation

**Assessment vs. Conference Standards:**
This is the more polished version suitable for publication. Minor improvements could include confidence intervals and final loss values.

---

## 3. Latent Space Visualizations

### 3.1 miwae_latents_by_sale_price.png

**Type:** 2D scatter plot of latent space colored by sale price deciles

**Dimensions:** Square aspect ratio, approximately 6" × 5"

**Axes:**
- **X-axis:** "z1" ranging from 0.0 to 0.5 (linear scale)
- **Y-axis:** "z2" ranging from -0.2 to 1.3 (linear scale)

**Title:** "Latent space (z1 vs z2) colored by sale-price deciles"

**Data Points:**
- Approximately 90,000 points (properties with observed sale prices)
- Each point represents the posterior mean latent representation of one property
- Points form a dense, elongated diagonal distribution from lower-left to upper-right

**Color Encoding:**
- **Colormap:** Viridis (perceptually uniform, purple → blue → green → yellow)
- **Colorbar:** Positioned on right side, labeled "Sale-price decile (0 = lowest, 9 = highest)"
- **Scale:** Discrete integer values 0-9 representing sale price deciles
- **Purple (decile 0):** Lowest-priced properties, concentrated in lower-left region (z1 ≈ 0.05-0.15, z2 ≈ -0.2-0.1)
- **Yellow (decile 9):** Highest-priced properties, concentrated in upper-right region (z1 ≈ 0.3-0.5, z2 ≈ 0.5-1.3)
- **Gradient:** Smooth progression from purple → green → yellow along the main diagonal axis

**Visual Pattern:**
- Clear diagonal gradient suggesting one principal component (combination of z1 and z2) captures price variation
- Main axis of variation runs from lower-left to upper-right at approximately 45-60 degrees
- Dense core region with deciles 3-7 (middle-priced properties)
- Outliers visible in upper portion (z2 > 1.0), mostly high-price yellow points
- Some purple low-price outliers scattered in upper-left region

**Point Density:**
- Highest density in center-bottom region (z1 ≈ 0.15-0.25, z2 ≈ 0.0-0.3)
- Density decreases toward edges
- Sparse regions in corners

**Statistical Interpretation:**
- Approximately linear relationship between latent position and price decile
- Suggests 1-2 latent dimensions primarily encode value/size continuum
- Third latent dimension (z0, not shown) likely captures orthogonal variation

**Grid:** White background, light gray gridlines at regular intervals

**Assessment vs. Conference Standards:**
Good visualization but caption should specify:
- Number of points plotted
- How deciles were computed (boundaries)
- Which latent dimensions (z1, z2 correspond to which original z0, z1, z2)
- Why these two dimensions were selected (variance explained)

---

### 3.2 miwae_latents_by_building_class.png

**Type:** 2D scatter plot of latent space colored by building class (categorical)

**Dimensions:** Approximately 6" × 12" (tall format to accommodate large legend)

**Axes:**
- **X-axis:** "z1" ranging from approximately -0.6 to 0.4 (linear scale)
- **Y-axis:** "z2" ranging from approximately -0.6 to 1.0 (linear scale)

**Title:** "Latent space (z1 vs z2) colored by building class"

**Subtitle:** (Below title area) "Building class (truncated)" with extensive legend showing building class codes

**Data Points:**
- Tens of thousands of points representing properties
- Points form two distinct clusters/lobes:
  1. **Lower-left cluster:** Centered around (z1 ≈ -0.4, z2 ≈ -0.3)
  2. **Upper-right cluster:** Centered around (z1 ≈ 0.1-0.2, z2 ≈ 0.3-0.5)
- Dense diagonal band connecting the two clusters

**Color Encoding:**
- **Over 100 building class codes** (e.g., Z3, D3, C7, R4, M1, etc.)
- **Multiple colors:** Each building class assigned a distinct color from a categorical palette
- **Legend:** Massive legend with ~100+ entries positioned at top, showing class codes like:
  - Z3, D3, C7, R4 (residential classes)
  - M1, M4, D4 (mixed-use/commercial)
  - L2, B2 (various)
  - Many others (E1, I1, Z5, L2, F1, V2, RM, G0, P2, G5, E4, H7, K7, J2, M3, C6, M2, S0, K6, W6, H3, W7, H9, J9, R, H5, W1, HB, A7, L3, U4, E2, Z3, H1, E7, Q2, J3, P3, Z2, J8, I7, G9, A3, I4, W8, J6, F2, U6, U1, H4, HS, HR, A0, W4, Z4, P6, G3, G8, N2, J7, GW, HH, P8, Y6, N4, N9, T9, F9)

**Visual Pattern:**
- **Two distinct groupings visible despite color complexity:**
  1. Lower-left lobe appears to contain different building class composition than upper-right
  2. The two lobes likely correspond to distinct property types (e.g., residential vs. commercial/mixed)
- **Striations:** Even with many colors, visible striations/bands suggest building classes are not randomly distributed
- **Separation:** Some building classes cluster tightly (small variance), others spread across latent space

**Technical Challenge:**
- Too many categories for effective color discrimination
- Legend overwhelms the plot
- Difficult to identify specific classes in the scatter

**Grid:** Light gray gridlines

**Assessment vs. Conference Standards:**
This figure has **significant issues** for conference publication:

**Problems:**
1. **Overplotting:** 100+ colors makes individual classes indistinguishable
2. **Legend dominates:** Legend is larger than the plot itself
3. **No grouping:** Building classes should be aggregated into meaningful categories
4. **Missing information:** No indication of which classes are most common
5. **Poor color palette:** Categorical palette unsuitable for this many categories

**Recommended Improvements for Conference Standards:**
1. **Aggregate building classes** into 5-10 meaningful categories (e.g., "1-2 Family Residential", "Multi-family", "Commercial", "Mixed-use", "Industrial", "Institutional")
2. **Use fewer, distinguishable colors** with clear legend
3. **Add density contours** for major categories
4. **Consider faceting** by category instead of overplotting
5. **Show representative classes only** with "Other" category
6. **Add statistical annotation:** Percentage of each category, sample sizes

This is one of the weakest figures in the collection and would likely receive critical reviewer feedback at a top conference.

---

### 3.3 latent_space_price.png

**Type:** Enhanced 2D scatter plot with density contours and detailed annotations

**Dimensions:** Square aspect ratio, approximately 8" × 8" (full width for presentation)

**Title:** "Latent Space by Price"

**Subtitle Box (upper-left):** "Contours: Density Iso-lines"

**Axes:**
- **X-axis:** Labeled with complex interpretable feature combination: "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)" - horizontal axis labeled "(z3)"
- **Y-axis:** Labeled with complex interpretable feature combination: "(24% Bldg Sales Vol, 20% Avg Cumul Sales, 17% Recent Trend)" - vertical axis labeled "(z2)"

**Footnote:** "Latent dimensions selected by Variance: z3 (50%) + z1 (30%)"

**Data Visualization:**
1. **Scatter points:** ~100,000 semi-transparent points colored by sale price
2. **Density contours:** Black curved isolines showing regions of high point density, with 5-6 concentric elliptical contours centered on the main distribution

**Color Encoding:**
- **Colormap:** Viridis (purple → teal → green → yellow)
- **Colorbar (right side):** Labeled "Sale Price" with USD values in exponential notation
- **Price range:** $100K (purple, bottom) to $250.0M (yellow, top)
- **Scale:** Appears to be log-scale given exponential progression: $100K → $250K → $500K → $1M → $2.5M → $5M → $10M → $25M → $50M → $100M → $250M

**Visual Pattern:**
- **Main distribution:** Dense elliptical cloud centered around (z3 ≈ -0.3, z2 ≈ 0.3)
- **Elongation:** Primary axis runs from lower-left to upper-right (similar to previous plots)
- **Color gradient:** Clear progression from purple (low price) in lower-left to yellow (high price) in upper-right
- **Density contours:** Roughly elliptical, tilted ~45-60 degrees
- **Outliers:**
  - High-price yellow outliers scattered in upper-right region
  - Some low-price purple outliers in lower-left
  - Light blue/green mid-price properties fill the central bulk

**Interpretability Annotations:**
The axis labels provide **feature attributions** showing which original features contribute to each latent dimension:
- **z3 (horizontal):** 26% Recent Trend + 19% Bldg Sales Vol + 16% Avg Cumul Sales
- **z2 (vertical):** 24% Bldg Sales Vol + 20% Avg Cumul Sales + 17% Recent Trend
Both axes capture similar features with different weights, explaining the diagonal gradient

**Variance Explained:**
- z3 explains 50% of latent variance
- z1 explains 30% of latent variance
- Total: 80% variance captured in 2D projection

**Background:** Clean white

**Assessment vs. Conference Standards:**
This is a **high-quality publication-ready figure** with several strengths:

**Strengths:**
1. Density contours add information beyond scatter alone
2. Interpretable axis labels (feature attributions)
3. Variance explained documented
4. Appropriate log-scale colorbar for skewed price distribution
5. Clear visual encoding

**Minor Improvements:**
1. Specify method used for feature attribution (SHAP? PCA loadings?)
2. Add sample size to caption
3. Clarify if percentages are absolute contributions or relative importance
4. Consider adding marginal density histograms
5. Note: "$250.0M" should be "$250M" for consistency

This figure would be suitable for ICML/NeurIPS with minor caption improvements.

---

### 3.4 latent_space_bldg.png

**Type:** Enhanced 2D scatter plot with density contours, colored by building class (aggregated)

**Dimensions:** Square aspect ratio, approximately 8" × 8"

**Title:** "Latent Space by Building Class"

**Subtitle Box (upper-left):** "Contours: Density Iso-lines"

**Axes:**
- **X-axis:** "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"
- **Y-axis:** "(24% Bldg Sales Vol, 20% Avg Cumul Sales, 17% Recent Trend)"
- Same interpretable feature attributions as latent_space_price.png

**Data Visualization:**
1. **Scatter points:** ~100,000 points colored by aggregated building class categories
2. **Density contours:** Black curved isolines (5-6 contours) showing overall density distribution

**Color Encoding - Building Classes (Legend, right side):**
Approximately 17 major building class categories using distinct colors:
1. **Elevator Apartments** (dark purple)
2. **Walk-up Apartments** (orange)
3. **Mixed Residential** (light blue)
4. **1-2 Family Houses** (green)
5. **2 Family Houses** (dark green)
6. **Retail/Stores** (pink)
7. **Office Buildings** (red)
8. **Hotels** (brown)
9. **Lofts** (light green)
10. **Warehouses** (yellow-green)
11. **Factories** (cyan)
12. **Garages** (blue-gray)
13. **Healthcare** (magenta)
14. **Entertainment** (olive)
15. **Religious** (purple-pink)
16. **Nursing/Asylums** (teal)
17. **Recreation (Indoor)** (dark blue)
18. **Educational** (yellow)
19. **Vacant Land** (light gray)
20. **Miscellaneous** (dark gray)

**Visual Pattern:**
- **Two distinct lobes:**
  1. **Lower-left cluster** (z3 ≈ -0.5, z2 ≈ -0.3): Dominated by residential classes (walk-ups, 1-2 family in orange/green)
  2. **Upper-right cluster** (z3 ≈ 0.2, z2 ≈ 0.4): Mixed composition with more elevator apartments, offices, retail
- **Diagonal band:** Transition zone showing mix of all categories
- **Clear separation:** Residential classes (especially 1-2 family houses and walk-ups) cluster differently from commercial/mixed-use
- **Density contours:** Cover the central bulk, cutting through both lobes

**Specific Class Distributions:**
- **Orange (Walk-up Apartments):** Concentrates in lower-left lobe with some presence in upper lobe
- **Dark blue (Elevator Apartments):** Scattered but more prevalent in upper-right
- **Green shades (1-2 Family, 2 Family):** Strong presence in lower-left cluster
- **Commercial classes (Retail, Office, Hotels):** More scattered, slight tendency toward upper-right
- **Rare classes (Healthcare, Religious, Educational):** Sparse, scattered outliers

**Striations visible:** Despite many categories, clear spatial organization suggests latent space encodes building type structure

**Background:** Clean white

**Footnote:** "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"

**Assessment vs. Conference Standards:**
**Major improvement over miwae_latents_by_building_class.png!** This aggregated version is much more suitable for publication:

**Strengths:**
1. Reasonable number of categories (~17-20 vs. 100+)
2. Clear spatial patterns visible
3. Density contours provide context
4. Consistent with latent_space_price.png in style

**Remaining Issues:**
1. Still challenging to distinguish 20 colors
2. Should show **percentages** or **counts** for each category in legend
3. Could benefit from **highlighting top 5-7 categories** and grouping others as "Other"
4. Missing **statistical tests** for separation (e.g., silhouette scores, cluster purity)

**Recommended Further Improvements:**
1. Reduce to 8-10 categories by grouping
2. Add inset bar chart showing category frequencies
3. Compute and report cluster separation metrics
4. Consider faceted plots showing each category separately

This would be acceptable at a conference but would benefit from the improvements above for top-tier publication.

---

### 3.5 latent_space_density.png

**Type:** 2D hexagonal binning (hexbin) plot showing point density in latent space

**Dimensions:** Square aspect ratio, approximately 8" × 8"

**Title:** "Latent Space Density"

**Axes:**
- **X-axis:** "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"
- **Y-axis:** "(24% Bldg Sales Vol, 20% Avg Cumul Sales, 17% Recent Trend)"

**Visualization Method:**
- **Hexagonal binning:** Space divided into hexagonal tiles
- **Each hexagon:** Represents a region of latent space, colored by number of properties in that region

**Color Encoding:**
- **Colormap:** Grayscale (white → black) with subtle gradation
- **Colorbar (right side):** Labeled "Count" with **logarithmic scale**
  - Range: 10⁰ (1 property) to 10³ (1000 properties)
  - Intermediate values: 10¹ (10), 10² (100)
- **White hexagons:** Regions with very few properties (1-10)
- **Light gray:** Moderate density (10-100)
- **Dark gray to black:** High density (100-1000+)

**Visual Pattern:**
- **Dense core:** Black hexagonal region in center-lower portion of plot, roughly elliptical
- **Primary dense region:** Centered around (z3 ≈ -0.3, z2 ≈ 0.2)
- **Secondary dense region:** Extends diagonally toward upper-right
- **Density gradient:** Gradual decrease from dark core to light periphery
- **Sparse regions:** White/very light hexagons in corners and edges
- **Elongated shape:** Main density axis runs lower-left to upper-right (consistent with previous plots)

**Density Distribution:**
- **Highest density:** Central hexagons contain 500-1000+ properties each
- **Medium density:** Surrounding belt contains 50-500 properties per hexagon
- **Low density:** Outer regions contain 1-50 properties per hexagon
- **Empty regions:** Some hexagons have zero properties (appear as gaps)

**Hexagon Coverage:**
- Hexagons tile the entire plotting area
- Hexagon size chosen to balance resolution vs. statistical stability
- Approximately 1000-2000 hexagons total in the grid

**Background:** Light gray background, white gridlines

**Footnote:** "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"

**Assessment vs. Conference Standards:**
This is an **excellent supplementary figure** for understanding data distribution:

**Strengths:**
1. Log-scale colorbar appropriate for heavy-tailed density distribution
2. Hexbins reduce overplotting compared to scatter
3. Clear visualization of where most properties lie
4. Complements the colored scatter plots well

**Weaknesses:**
1. **Less informative** than scatter + contours for showing patterns
2. **No direct connection** to outcome (price) or categories (building class)
3. **Hexagon size not specified** - critical for reproducibility

**Role in Paper:**
- Best suited for **supplementary materials** rather than main text
- Useful for justifying claims about data coverage
- Could support discussion of model fit in dense vs. sparse regions

**Recommended Improvements:**
1. Specify hexagon size and number
2. Add percentage of total data in each density tier
3. Overlay model prior density contours to show coverage
4. Cross-reference with prediction error to show if errors concentrate in sparse regions

This figure is publication-quality for supplementary materials.

---

### 3.6 latent_marginals.png

**Type:** Multi-panel histogram showing marginal distributions of latent dimensions

**Dimensions:** Approximately 10" × 4" (wide format for 3 panels)

**Title:** "Latent Marginal Distributions"

**Layout:** Three side-by-side histograms

**Panel 1 (Left) - First Latent Dimension:**
- **X-axis label:** "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"
- **X-axis range:** -0.50 to 0.75
- **Y-axis:** "Density" ranging from 0 to ~3.5
- **Distribution shape:** Roughly symmetric with slight right skew
  - Peak around 0.0
  - Spread from -0.3 to 0.5
  - Approximately Gaussian with kurtosis close to 3
- **Histogram bars:** Blue fill, approximately 20-25 bins

**Panel 2 (Center) - Second Latent Dimension:**
- **X-axis label:** Similar feature composition (exact text not fully visible in description)
- **X-axis range:** -0.50 to 0.75
- **Y-axis:** "Density" ranging from 0 to ~2.5
- **Distribution shape:** Multimodal or heavy-tailed
  - Primary peak around -0.25
  - Secondary mode or heavy tail on right
  - More skewed than Panel 1
  - Spread from -0.5 to 0.6
- **Histogram bars:** Blue fill

**Panel 3 (Right) - Third Latent Dimension:**
- **X-axis label:** "(24% Bldg Sales Vol, 20% Avg Cumul Sales, 17% Recent Trend)"
- **X-axis range:** -0.50 to 0.75
- **Y-axis:** "Density" ranging from 0 to ~9
- **Distribution shape:** Highly concentrated, nearly unimodal
  - Very sharp peak around 0.1
  - Extremely concentrated (small variance)
  - Minimal spread (mostly between 0.0 and 0.3)
  - Tall peak indicates low variance explained by this dimension
- **Histogram bars:** Blue fill

**Visual Comparison Across Panels:**
- **Panel 1:** Moderate spread, symmetric
- **Panel 2:** Moderate spread, asymmetric/multimodal
- **Panel 3:** Narrow spread, very peaked (low variance)

**Grid:** Light gray dotted grid on white background

**Footnotes:**
- Below Panel 1: "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"
- Below Panel 3: "(24% Bldg Sales Vol, 20% Avg Cumul Sales, 17% Recent Trend)"

**Statistical Interpretation:**
- The three panels show different variance levels, consistent with PCA-like structure
- Panel 3's sharp peak suggests the third latent dimension captures less variation
- Panel 1's broader distribution suggests it captures primary variation
- None appear perfectly Gaussian, suggesting the Student-t prior was appropriate

**Assessment vs. Conference Standards:**
This is a **useful diagnostic figure** but has some issues:

**Strengths:**
1. Shows all marginal distributions for interpretability
2. Density scale (vs. count) makes distributions comparable
3. Feature attributions provide interpretability

**Weaknesses:**
1. **No overlay** of prior distribution for comparison
2. **No quantitative statistics** (mean, std, skewness, kurtosis)
3. **Unclear ordering:** Why this dimension order? Should be ordered by variance explained
4. **Missing key information:** Which panel corresponds to which z dimension (z0, z1, z2)?
5. **Inconsistent labeling:** Feature attributions differ between panels but not clearly distinguished

**Critical Missing Elements for Conference Publication:**
1. **Overlay Student-t mixture prior** to show prior vs. posterior
2. **Label as z0, z1, z2** clearly
3. **Sort by variance explained** (if PCA-like)
4. **Add summary statistics** to each panel
5. **Show marginal prior** densities for comparison

**Recommended Improvements:**
```
Panel titles: "z3 (50% variance)" | "z1 (30% variance)" | "z0 (20% variance)"
Overlay: Dashed line showing prior marginal distribution
Annotation: Mean ± std for each dimension
```

This figure needs significant enhancement for top-tier publication but is acceptable for supplementary materials.

---

### 3.7 latent_space_price_hexbin.png

**Type:** Hexagonal binning plot colored by mean sale price

**Dimensions:** Square aspect ratio, approximately 8" × 8"

**Title:** "Latent Space by Price"

**Subtitle box (upper-left):** "Opacity ∝ Density"

**Axes:**
- **X-axis:** "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"
- **Y-axis:** "(24% Bldg Sales Vol, 20% Avg Cumul Sales, 17% Recent Trend)"

**Visualization Method:**
- **Hexagonal binning:** Space divided into hexagonal tiles
- **Color:** Each hexagon colored by mean sale price of properties in that bin
- **Opacity/Alpha:** Hexagon opacity proportional to number of properties (density)

**Color Encoding:**
- **Colormap:** Viridis (purple → teal → yellow)
- **Colorbar (right):** Labeled "Mean Sale Price"
- **Price range:** $100K to $250.0M (likely log-scaled)
- **Gradation:** $100K, $250K, $500K, $1M, $2.5M, $5M, $10M, $25M, $50M, $100M, $250M

**Opacity Encoding:**
- **Transparent/faint:** Hexagons with few properties (low density)
- **Opaque/solid:** Hexagons with many properties (high density)
- This creates a visual hierarchy: high-confidence estimates (many samples) are bold, low-confidence (few samples) are faint

**Visual Pattern:**
- **Lower-left region:** Purple and dark blue hexagons (low prices $100K-$500K), high opacity (many properties)
- **Upper-right region:** Yellow-green hexagons (high prices $10M-$100M+), moderate opacity
- **Diagonal gradient:** Clear color progression from lower-left (purple) to upper-right (yellow)
- **Dense core:** Solid opaque hexagons in center showing teal/green colors ($1M-$5M range)
- **Sparse periphery:** Faint hexagons on edges

**Spatial-Price Relationship:**
- Lower-left: Low-priced properties cluster
- Upper-right: High-priced properties cluster
- Smooth transition along diagonal axis
- Confirms that latent space organizes properties by value/size

**Background:** White

**Footnote:** "(26% Recent Trend, 19% Bldg Sales Vol, 16% Avg Cumul Sales)"

**Assessment vs. Conference Standards:**
This is a **clever visualization** combining density and mean price:

**Strengths:**
1. **Dual encoding:** Color (mean price) + opacity (density) conveys two dimensions of information
2. **Statistical rigor:** Opacity indicates confidence (sample size)
3. **Reduces noise:** Averaging within hexagons reduces scatter noise
4. **Clear pattern:** Price gradient is clearer than in scatter plot

**Weaknesses:**
1. **Hexagon size not specified**
2. **No error bars** or variance indication within hexagons
3. **Binning artifacts:** Hexagon boundaries create artificial edges
4. **Missing information:** Number of hexagons, minimum samples per hexagon
5. **Opacity mapping unclear:** What's the exact relationship between density and opacity?

**Comparison to latent_space_price.png:**
- **Hexbin version:** Better for showing aggregate patterns, less overplotting
- **Scatter version:** Better for showing individual-level variation and outliers
- **Recommendation:** Include both in paper (hexbin in main text, scatter in supplement)

**Recommended Improvements:**
1. Specify hexagon parameters (size, min samples)
2. Add error bars or variance overlay
3. Clarify opacity scale (linear? log?)
4. Add sample size annotation showing hexagons with <10, <50, <100 samples
5. Consider overlay showing hexagons with too few samples for reliable estimates

This figure is **publication-ready with minor caption improvements**.

---

### 3.8 latent_space_price_decile_orig.png

**Type:** 2D scatter plot colored by price deciles (original simple version)

**Dimensions:** Square aspect ratio, approximately 6" × 6"

**Title:** "Latent space (z3 vs z1) colored by sale-price deciles\n(Original Simple Version)"

**Axes:**
- **X-axis:** "z3" ranging from approximately -1.0 to 0.0
- **Y-axis:** "z1" ranging from 0.0 to 0.5
- **NOTE:** Different latent dimensions than other plots (z3 vs z1 instead of z1 vs z2)

**Data Points:**
- Thousands of points representing properties
- Dense elliptical distribution

**Color Encoding:**
- **Colormap:** Viridis (purple → yellow)
- **Colorbar (right):** "Sale-price decile (0=lowest, 9=highest)"
- **Scale:** 0 to 9 (discrete deciles)

**Visual Pattern:**
- **Main distribution:** Elliptical cloud centered around (z3 ≈ -0.5, z1 ≈ 0.2)
- **Color gradient:** Clear progression from purple (low price, left side) to yellow (high price, upper-right)
- **Diagonal orientation:** Gradient runs from lower-left toward upper-right
- **Overlap:** Significant overlap of different deciles, suggesting these dimensions don't perfectly separate price levels

**Comparison to miwae_latents_by_sale_price.png:**
- **Different axes:** This shows z3 vs z1, while miwae version shows z1 vs z2
- **Simpler styling:** No axis interpretations, no density contours
- **Different orientation:** Distribution oriented differently on the page

**Label:** "(Original Simple Version)" suggests this was an early iteration

**Assessment vs. Conference Standards:**
This is a **draft/preliminary version**:

**Issues:**
1. No axis interpretability (feature attributions missing)
2. Less informative axis choice (different z dimensions)
3. No density contours
4. Simple title lacks context
5. Missing variance explained
6. No clear advantage over the enhanced versions

**Recommendation:**
- **Do not use in publication**
- Replaced by latent_space_price.png which is superior
- Useful only for showing progression of figure development

---

### 3.9 latent_space_bldg_orig.png

**Type:** 2D scatter plot colored by building class (original version with 100+ categories)

**Dimensions:** Very tall format, approximately 6" × 14" (to fit massive legend)

**Title:** "Latent space (z1 vs z2) colored by building class\n(Original Simple Version)"

**Subtitle:** "Building class (truncated)"

**Axes:**
- **X-axis:** "z1" (appears to be different scale than enhanced version)
- **Y-axis:** "z2" (appears to be different scale than enhanced version)

**Data Points:**
- Points form diagonal band with two main lobes

**Color Encoding:**
- **100+ building class codes** each with unique color
- **Massive legend** occupying ~60% of vertical space, listing codes like: D4, M1, C6, V1, Q1, C5, C1, D1, B9, A9, L8, C9, C7, C4, C0, M9, A5, B3, B1, S2, A4, C3, D0, K1, O7, O3, M0, O2, D7, C2, S9, N9, G9, G6, K9, G1, S5, G7, D6, S4, Z9, R, L1, K2, G2, A1, E9, O4, D9, B2, I5, M1, A3, I4, Q9, O8, J6, Y8, V3, I9, P7, O6, O1, Y8, V3, I9, P7, O6, O1, U2, F9 (and many more)

**Visual Pattern:**
- Extremely difficult to discern individual categories due to color overload
- Two main spatial clusters visible despite color chaos

**Assessment vs. Conference Standards:**
**This is the problematic original version** that should NOT be used:

**Critical Flaws:**
1. ❌ 100+ colors completely uninterpretable
2. ❌ Legend larger than plot
3. ❌ No aggregation or grouping
4. ❌ No statistical information
5. ❌ Violates basic data visualization principles

**This figure was correctly replaced by latent_space_bldg.png**

**Recommendation:**
- **Never use in publication**
- Example of what NOT to do
- Useful only for teaching visualization principles

---

## 4. Residual Analysis

### 4.1 miwae_residuals_histogram.png

**Type:** Histogram with probability density on y-axis

**Dimensions:** Standard single-column width, approximately 5" × 3.5"

**Title:** "Residuals in log-price space"

**Axes:**
- **X-axis:** "Residual (log_y_true - log_y_pred)" ranging from approximately -15 to +10
- **Y-axis:** "Density" ranging from 0 to 0.7

**Histogram Specifications:**
- **Color:** Blue fill with darker blue borders
- **Bins:** Approximately 15-20 bins
- **Range:** Residuals span from about -12 to +8 log units

**Distribution Shape:**
- **Strongly peaked** around 0 (mode near 0)
- **Highly asymmetric:**
  - **Right tail** extends to +8 log units (moderate positive residuals)
  - **Left tail** extends to -15 log units (extreme negative residuals, underpredictions)
- **Peak density:** ~0.7 at residual ≈ 0
- **Kurtosis:** Visibly leptokurtic (heavy-tailed), much heavier tails than Gaussian
- **Median:** Appears to be near 0 (unbiased on average)

**Tail Behavior:**
- **Left tail (negative residuals):** Long tail with non-negligible mass at -10 to -15
  - Indicates systematic underprediction for some properties
  - Extreme negative residuals suggest model predicts much higher prices than actual for some sales
- **Right tail (positive residuals):** Shorter but still heavy, extends to +8
  - Some overpredictions but less extreme
- **Central 90%:** Roughly -2 to +2 log units

**Background:** White with light gray gridlines

**Assessment vs. Conference Standards:**
This figure is **acceptable but minimal**:

**Strengths:**
1. Clear visualization of heavy tails
2. Density scale makes it comparable across datasets
3. Shows the distribution shape clearly

**Critical Weaknesses:**
1. ❌ **No reference distribution** (e.g., Gaussian overlay with same mean/variance)
2. ❌ **No quantitative statistics** annotated (kurtosis, skewness, percentiles)
3. ❌ **No indication of sample size**
4. ❌ **Missing context:** What's an acceptable residual magnitude for log-price?
5. ❌ **No outlier thresholds** marked

**What's Missing for Conference Standards:**
1. **Overlay Gaussian reference** with matched variance (dashed line)
2. **Annotate kurtosis** (report states ≈9.27)
3. **Mark percentiles** (5th, 25th, 50th, 75th, 95th)
4. **Add text box** with summary statistics:
   ```
   Mean: 0.XX
   Median: 0.XX
   Std: X.XX
   Kurtosis: 9.27
   N = XXXXX
   ```
5. **Mark ±2σ, ±3σ bounds**

**Comparison to Other Histogram Versions:**
This appears to be the simplest version. Other files (residuals_hist_normal.png, residuals_hist_references.png, residuals_hist_simple.png) likely have enhancements.

---

### 4.2 miwae_residuals_qq.png

**Type:** Quantile-Quantile plot comparing residuals to Normal distribution

**Dimensions:** Standard square aspect ratio, approximately 5" × 5"

**Title:** "QQ-plot of log-price residuals vs Normal"

**Axes:**
- **X-axis:** "Theoretical quantiles" ranging from -4 to +4 (standard Normal quantiles)
- **Y-axis:** "Ordered Values" ranging from approximately -15 to +12 (observed residual quantiles)

**Plot Elements:**
1. **Reference line (red):** Diagonal line from lower-left to upper-right representing perfect Normal distribution, with slope equal to residual standard deviation
2. **Data points (blue):** Ordered residuals plotted against expected Normal quantiles
   - Points form a distinctive S-curve

**Visual Pattern - Classic Heavy-Tail Signature:**
- **Lower tail (left side):**
  - Blue points **far below** red line from x ≈ -4 to x ≈ -1
  - Extreme negative outlier around (x=-3.5, y=-15)
  - Indicates **left tail is heavier** than Normal (more extreme negative residuals)

- **Middle section:**
  - Points roughly follow red line from x ≈ -1 to x ≈ +1
  - Suggests **central 68%** approximates Normal

- **Upper tail (right side):**
  - Blue points **far above** red line from x ≈ +1 to x ≈ +4
  - Extreme positive outliers reaching y ≈ +12
  - Indicates **right tail is heavier** than Normal (more extreme positive residuals)

**Quantitative Deviations:**
- At x = -3 (99.7th percentile of Normal), Normal predicts y ≈ -6, but observed y ≈ -12 (2× heavier tail)
- At x = +3 (99.7th percentile of Normal), Normal predicts y ≈ +6, but observed y ≈ +12 (2× heavier tail)

**Interpretation:**
- **S-curve shape** is diagnostic of **leptokurtic distribution** (heavy tails, peaked center)
- Consistent with report's claim of kurtosis ≈ 9.27 (vs. 3 for Normal)
- Suggests Student-t or other heavy-tailed distribution would be more appropriate

**Background:** White grid

**Assessment vs. Conference Standards:**
This is a **good diagnostic plot** but needs enhancements:

**Strengths:**
1. Clear visualization of tail heaviness
2. Reference line properly scaled
3. S-curve pattern is unmistakable

**Weaknesses:**
1. ❌ **No confidence band** around reference line (e.g., 95% pointwise envelope)
2. ❌ **No kurtosis value** annotated
3. ❌ **No quantitative tail comparison** (e.g., "tails 2× heavier than Normal")
4. ❌ **Missing sample size**
5. ❌ **No comparison to Student-t reference**

**Conference-Quality Enhancements:**
1. Add **gray confidence envelope** showing acceptable deviation from Normal
2. **Overlay Student-t QQ line** with degrees of freedom ≈ 4 for comparison
3. **Annotate kurtosis** = 9.27 in corner
4. Add text box:
   ```
   Excess Kurtosis: 6.27
   Shapiro-Wilk: p < 0.001
   N = XXXXX
   ```
5. Mark specific quantiles (±1σ, ±2σ, ±3σ points)

**Comparison to Other QQ Versions:**
Files residuals_qq.png, residuals_qq_simple.png likely show iterations of this plot.

---

### 4.3 residuals_hist_normal.png

**Type:** Histogram with overlaid Normal reference distribution

**Dimensions:** Approximately 6" × 4"

**Title:** "Residuals vs Normal"

**Subtitle:** "Log-transformed prices"

**Axes:**
- **X-axis:** "Residual" ranging from -10 to +10
- **Y-axis:** "Density" ranging from 0 to 2.0

**Plot Elements:**
1. **Histogram (green bars):** Observed residuals labeled as "Observed"
   - Approximately 12-15 bins
   - Tall central peak around 0
   - Short tails extending to ±10

2. **Normal reference curve (dashed gray line):** Labeled as "Normal (MLE fit)"
   - Fitted Normal distribution with maximum likelihood parameters
   - Much flatter and wider than observed distribution
   - Peak around 0.35-0.4 density

**Visual Comparison:**
- **Observed distribution (green):**
  - **Much taller peak** (~1.7 density) than Normal (~0.35)
  - **Narrower center**
  - **Heavier tails** (bins visible at ±8 where Normal predicts near-zero)

- **Normal reference (gray):**
  - **Flatter peak**
  - **Wider spread**
  - **Lighter tails**

**Legend:** Upper-right corner
- Green solid: "Observed"
- Gray dashed: "Normal (MLE fit)"

**Interpretation:**
- Clear visual demonstration that residuals are **leptokurtic** (peaked + heavy-tailed)
- The observed distribution has **much more mass** at center and tails, less in shoulders
- This is exactly the pattern expected for kurtosis > 3

**Background:** White with light gray gridlines

**Assessment vs. Conference Standards:**
This is an **excellent pedagogical figure**:

**Strengths:**
1. ✅ **Direct comparison** to Normal reference
2. ✅ **MLE fit** ensures fair comparison (not arbitrary Normal)
3. ✅ **Clear visual contrast** between observed and reference
4. ✅ **Good color choice** (green vs. gray is distinguishable)

**Minor Weaknesses:**
1. **No quantitative statistics** annotated
2. **No confidence bands** on Normal reference
3. **Missing key values** (kurtosis, sample size, fitted μ and σ)

**Recommended Enhancements:**
Add text box:
```
Observed:
  Kurtosis: 9.27
  Std: X.XX
  N = XXXXX

Normal Fit:
  μ = 0.XX
  σ = X.XX
  Kurtosis: 3.00
```

**This figure is publication-ready with minor caption additions.**

---

### 4.4 residuals_hist_references.png

**Type:** Histogram with overlaid Student-t reference distribution

**Dimensions:** Approximately 7" × 5"

**Title:** "Residuals vs References"

**Subtitle:** Formula shown: r = log(y_true) - log(y_pred)

**Axes:**
- **X-axis:** "Residual" ranging from -2 to +2
- **Y-axis:** "Density" ranging from 0 to 3.0

**Plot Elements:**
1. **Histogram (green bars):** Observed residuals
   - 7-8 wide bins
   - Concentrated in range [-1, +1]
   - Peak density ~1.6 at residual ≈ 0

2. **Student-t reference curve (solid red line):** Labeled as "Best Fit (Student-t, ν=1.1)"
   - Heavy-tailed distribution fit
   - Degrees of freedom ν = 1.1 (very heavy tails, close to Cauchy)
   - Peak around 2.7 density
   - Long tails extending beyond ±2

3. **Confidence intervals (dashed and dotted vertical lines):**
   - **50% Interval (dotted):** Inner pair of vertical lines
   - **95% Interval (dashed):** Outer pair of vertical lines
   - Symmetric around 0

**Legend:** Upper-right corner
- Green bars: "Observed residuals"
- Red solid: "Best Fit (Student-t, ν=1.1)"
- Gray dashed: "95% Interval"
- Gray dotted: "50% Interval"

**Visual Fit:**
- **Red curve** fits the observed histogram reasonably well
- Student-t with ν=1.1 captures the peaked center and heavy tails
- Histogram shows slight asymmetry (left skew) not captured by symmetric Student-t

**Confidence Intervals:**
- **50% interval:** Appears to span approximately [-0.5, +0.1] (slightly asymmetric, confirming left skew)
- **95% interval:** Appears to span approximately [-1.0, +0.5]

**Background:** White

**Assessment vs. Conference Standards:**
This is an **excellent model-checking figure**:

**Strengths:**
1. ✅ **Appropriate reference** (Student-t, not Normal)
2. ✅ **Fitted degrees of freedom** shown (ν=1.1)
3. ✅ **Confidence intervals** provide context
4. ✅ **Clear formula** for residuals
5. ✅ **Honest assessment** - shows that even Student-t doesn't perfectly fit

**Interesting Finding:**
- **ν = 1.1** is extremely low (close to Cauchy distribution with ν=1)
- Suggests **extraordinarily heavy tails**
- This is critical information for the paper

**Weaknesses:**
1. **Range limited to [-2, +2]** hides extreme tails shown in other plots
2. **No goodness-of-fit statistic** (e.g., KS test, Anderson-Darling)
3. **No comparison to Normal** in same plot
4. **Sample size not shown**
5. **Fit method not specified** (MLE? moment matching?)

**Critical Question:**
Why does miwae_residuals_histogram.png show range [-15, +10] but this shows [-2, +2]? Are these different subsets of data?

**Recommended Enhancements:**
1. Show full range [-15, +10] OR justify [-2, +2] subset
2. Add goodness-of-fit p-value
3. Include Normal overlay for comparison
4. Specify fitting method
5. Add caption explaining why ν=1.1 is so low

**This is a strong figure but needs methodological clarification.**

---

### 4.5 residuals_hist_simple.png

**Type:** Simple histogram (no overlay)

**Dimensions:** Approximately 5" × 4"

**Title:** None visible (very simple version)

**Axes:**
- **X-axis:** Unlabeled, ranging from approximately -4 to +12
- **Y-axis:** Unlabeled, ranging from 0 to ~1.7

**Histogram:**
- **Color:** Green fill
- **Bins:** ~8-10 wide bins
- **Shape:** Strongly peaked at 0, heavy right tail extending to +10

**Visual Pattern:**
- Similar distribution to other histograms but simpler presentation
- Clear leptokurtic shape

**Assessment:**
This is a **preliminary/draft version**:

**Issues:**
1. ❌ No title
2. ❌ No axis labels
3. ❌ No reference distribution
4. ❌ No statistics

**Recommendation:**
- **Do not use in publication**
- Superseded by residuals_hist_normal.png and residuals_hist_references.png

---

### 4.6 residuals_qq.png

**Type:** Quantile-Quantile plot (standardized residuals)

**Dimensions:** Square aspect ratio, approximately 5" × 5"

**Title:** "QQ Plot"

**Axes:**
- **X-axis:** "Theoretical Quantiles" ranging from -4 to +4
- **Y-axis:** "Standardized Residuals" ranging from -4 to +5

**Plot Elements:**
1. **Reference line (red dashed):** Diagonal from (-5, -5) to (+5, +5)
2. **Data points (blue):** Standardized residuals vs. Normal quantiles
   - Form characteristic S-curve

**Visual Pattern:**
- **Lower tail:** Points below line (heavier left tail than Normal)
- **Center:** Points approximately on line
- **Upper tail:** Points above line (heavier right tail than Normal)
- S-curve less extreme than miwae_residuals_qq.png

**Note:** "Standardized Residuals" suggests residuals divided by standard deviation

**Assessment:**
Similar to miwae_residuals_qq.png but with standardized residuals.

**Differences:**
- Uses "Standardized Residuals" (divided by σ)
- Slightly cleaner styling
- Range differs from miwae version

**Recommendation:**
Choose ONE QQ plot for publication (either this or miwae_residuals_qq.png). This version is cleaner.

---

### 4.7 residuals_qq_simple.png

**Type:** QQ plot (original simple version)

**Dimensions:** Square aspect ratio, approximately 5" × 5"

**Title:** "QQ-plot: Residuals vs Normal\n(Original Simple Version)"

**Axes:**
- **X-axis:** "Theoretical Quantiles (Normal)" ranging from -4 to +4
- **Y-axis:** "Ordered Residuals" ranging from -4 to +12

**Plot Elements:**
1. **Reference line (red):** Diagonal
2. **Data points (blue):** Dense S-curve showing heavy tails

**Visual Pattern:**
- Same S-curve pattern as other QQ plots
- Extreme upper tail reaching +12
- Lower tail reaching -4

**Label:** "(Original Simple Version)"

**Assessment:**
This is the **original draft** version:

**Recommendation:**
- **Do not use in publication**
- Superseded by cleaner versions (residuals_qq.png or miwae_residuals_qq.png)

---

### 4.8 residuals_spatial_map.png

**Type:** Geographic map with point markers colored by residuals

**Dimensions:** Tall format, approximately 4" × 8" (vertical)

**Title:** "Spatial Residual Map"

**Map Area:**
- **Geographic region:** Manhattan (New York City), showing street grid and neighborhood labels
- **Basemap:** Light gray OpenStreetMap layer showing streets, parks, water bodies
- **Neighborhoods labeled:** Teaneck, Cliffside Park, Fort Lee, Palisades Park, Ridgefield, Fairview, North Bergen, Weehawken, Guttenborg (West side - New Jersey), and East New York, Cuttenburg (labeled on map)

**Map Attribution:** "© OpenStreetMap contributors (C) CARTO" at bottom

**Data Points:**
- Several hundred points scattered across Manhattan
- Each point represents a property with residual information

**Color Encoding:**
- **Colormap:** Diverging Red-White-Blue (RdBu)
- **Colorbar (right side):** "Mean Residual" ranging from -1.00 to +1.00
  - **Dark blue:** Residual = -1.0 (underprediction by 1 log unit, actual price ~2.7× higher than predicted)
  - **White:** Residual = 0 (accurate prediction)
  - **Dark red:** Residual = +1.0 (overprediction by 1 log unit, actual price ~2.7× lower than predicted)

**Spatial Pattern:**
- **Mixed spatial distribution:** Red and blue points are interspersed
- **Some clustering visible:**
  - Concentration of **red points** (overpredictions) in certain areas, particularly in lower Manhattan and parts of midtown
  - Concentration of **blue points** (underpredictions) in other areas, including some waterfront areas
  - **Central Manhattan:** Mix of both red and blue
  - **Upper Manhattan:** Sparser coverage, mixed colors

- **No strong continuous spatial gradient** (unlike a phenomenon with smooth spatial autocorrelation)
- **Point density varies:** Denser in central/lower Manhattan, sparser in upper Manhattan

**Marker Properties:**
- **Size:** Uniform (doesn't encode additional information)
- **Shape:** Circular
- **Edge:** Slight dark border for visibility

**Interpretation:**
- **Spatial bias exists:** Model systematically over/underpredicts in certain locations
- **Not random:** If residuals were spatially random, we'd see even mixing everywhere
- **Moderate effect size:** Most residuals within ±0.5 (factor of ~1.6×)
- **Suggests missing spatial covariates:** Neighborhood effects, proximity to amenities, etc. not fully captured

**Assessment vs. Conference Standards:**
This is a **critical diagnostic figure** showing model limitations:

**Strengths:**
1. ✅ **Direct visualization** of spatial bias
2. ✅ **Appropriate diverging colormap** (red-blue)
3. ✅ **Proper map attribution**
4. ✅ **Clear colorbar** with interpretable units

**Weaknesses:**
1. ❌ **No statistical test** for spatial autocorrelation (e.g., Moran's I)
2. ❌ **No neighborhood boundaries** overlaid
3. ❌ **Point overlap** makes some areas hard to interpret
4. ❌ **No density/aggregation** (hard to see patterns in dense areas)
5. ❌ **Sample size not shown**
6. ❌ **No indication of property types** (are red points all commercial?)

**Critical Missing Analysis:**
1. **Moran's I statistic** and p-value for spatial autocorrelation
2. **Neighborhood-level aggregation** (show mean residual by neighborhood)
3. **Split by building class** to see if bias is type-specific
4. **Variogram** showing residual correlation vs. distance

**Recommended Enhancements:**
1. Add semi-transparent point overlay to show density
2. Compute and annotate Moran's I: "Spatial autocorrelation: I = X.XX, p < 0.001"
3. Add neighborhood boundaries with mean residual per neighborhood
4. Create faceted version by building class
5. Add second panel showing prediction variance (uncertainty)

**Conference Reviewer Likely Feedback:**
"The spatial map suggests systematic geographic bias, but the authors provide no quantitative assessment of spatial autocorrelation. Please report Moran's I and discuss whether including spatial random effects would improve the model."

**This figure is publication-worthy but REQUIRES spatial autocorrelation statistics in the caption.**

---

### 4.9 residuals_by_bldg_class.png

**Type:** Point plot with error bars showing residuals by building class

**Dimensions:** Wide format, approximately 10" × 6"

**Title:** "Performance by Building Class"

**Subtitle:** "(Values in Log Space. Price ≥ $100K)"

**Axes:**
- **X-axis:** "Building Class" with 14 categories (text labels)
- **Y-axis:** "Mean Residual" ranging from -0.25 to 1.75

**Building Classes (left to right):**
1. Elevator Apartments
2. Walk-up Apartments
3. Mixed Residential
4. 1-2 Family Houses
5. 2 Family Houses
6. Retail/Stores
7. Office Buildings
8. Hotels
9. Lofts
10. Warehouses
11. Garages
12. Healthcare
13. Religious
14. Vacant Land

**Plot Elements:**
1. **Reference line (dashed black):** Horizontal line at y = 0 labeled "Zero Reference"
2. **Point estimates (teal circles):** Mean residual for each building class
3. **Error bars (vertical teal lines):** Confidence intervals (likely ±1 SE or 95% CI)

**Visual Pattern:**
- **Most classes near zero:** Categories 1-10 have mean residuals close to 0 (within ±0.5)
  - Elevator Apartments: ~+0.02
  - Walk-up Apartments: ~+0.30
  - Mixed Residential: ~+0.33
  - 1-2 Family Houses: ~+0.10
  - 2 Family Houses: ~+0.42
  - Retail/Stores: ~+0.25
  - Office Buildings: ~+0.10
  - Hotels: ~-0.20 (slight underprediction)
  - Lofts: ~+0.15
  - Warehouses: ~-0.03

- **Extreme bias for rare classes:**
  - **Garages:** ~+0.27
  - **Healthcare:** ~+0.67 (large positive bias - overpredicts by ~2×)
  - **Religious:** ~+1.30 (extreme positive bias - overpredicts by ~3.7×)
  - **Vacant Land:** ~+1.40 (extreme positive bias - overpredicts by ~4×)

**Error Bar Patterns:**
- **Small error bars** for common classes (Elevator, Walk-up, 1-2 Family) - high confidence
- **Large error bars** for rare classes (Healthcare, Religious, Vacant Land) - low confidence
- Error bar for Vacant Land extends from ~+0.9 to ~+1.7

**Interpretation:**
- Model performs **well on common residential and commercial** types
- Model **systematically overpredicts** for rare/unusual property types
- This suggests **sample size imbalance** in training data
- Rare classes may need special handling or stratified modeling

**Background:** White with light gray gridlines

**Assessment vs. Conference Standards:**
This is an **essential diagnostic figure**:

**Strengths:**
1. ✅ **Clear visualization** of class-specific bias
2. ✅ **Error bars** show uncertainty
3. ✅ **Zero reference line** aids interpretation
4. ✅ **Appropriate for log-space** interpretation

**Weaknesses:**
1. ❌ **No sample sizes** shown for each class
2. ❌ **Error bar definition unclear** (SE? 95% CI?)
3. ❌ **No statistical significance** indicators (e.g., stars for p < 0.05)
4. ❌ **No filtering justification** (why Price ≥ $100K?)
5. ❌ **Classes not ordered** (should order by sample size or bias magnitude)

**Critical Missing Information:**
- **Sample sizes:** Likely Religious and Vacant Land have very few samples, explaining extreme bias
- **Statistical tests:** Are Healthcare/Religious/Vacant Land significantly different from zero?
- **Decision threshold:** Should these classes be excluded or modeled separately?

**Recommended Enhancements:**
1. Add sample sizes below each class label: "Religious\n(n=42)"
2. Specify error bar type in caption: "Error bars show ±1 SE"
3. Add significance stars: * p<0.05, ** p<0.01, *** p<0.001
4. Order classes by sample size (left = most common)
5. Add vertical separator between common and rare classes
6. Consider log-scale or broken axis for extreme values
7. Add text box: "Classes with n < 100 shown in gray"

**Conference Reviewer Likely Question:**
"The authors show extreme bias for rare building classes (Religious, Vacant Land) but provide no sample sizes or plan to address this. Are these classes excluded from evaluation metrics? How does this affect real-world deployment?"

**This figure is publication-worthy but MUST include sample sizes.**

---

### 4.10 residuals_standardized_vs_pred.png

**Type:** Scatter plot with LOESS trend and confidence band

**Dimensions:** Wide format, approximately 8" × 5"

**Title:** "Residuals vs Prediction"

**Axes:**
- **X-axis:** "Predicted Price" with currency labels ($100K, $250K, $500K, $1M, $2.5M, $5M, $10M, $25M, $50M, $100M, $250M)
  - **Appears to be log-scale** on x-axis (evenly spaced on log scale)
- **Y-axis:** "Residual" ranging from -2 to +2 (log units)

**Subtitle:** "Residuals are standardized (r / σ)"

**Data Points:**
- **Thousands of points** (appears to be ~10,000-50,000)
- **Semi-transparent** to show density
- **Colors:**
  - Light blue/cyan: Most points
  - Light pink/salmon: Some points (different sign or magnitude?)

**Visual Elements:**
1. **Reference line (dashed black):** Horizontal at y = 0 (perfect prediction)

2. **LOESS trend line (solid red):** Smoothed conditional mean
   - Shows **systematic bias pattern**:
     - $100K-$250K: Residual ≈ +0.2 (overprediction)
     - $250K-$500K: Residual drops to ≈ -0.5 (underprediction)
     - $500K-$2M: Residual rises to ≈ 0 (unbiased)
     - $2M-$10M: Residual ≈ -0.2 (slight underprediction)
     - $10M+: Residual ≈ -0.1 to 0 (approximately unbiased)

3. **Confidence band (shaded pink):** ±1 standard deviation envelope
   - Band width varies with prediction:
     - Narrow ($1M-$5M): ±1.0
     - Wide at extremes ($100K): ±2.0, ($250M): ±1.3
   - Band is **not constant**, indicating heteroskedasticity

**Visual Patterns:**
- **Heteroskedasticity:** Clear variance increase at low and high price ranges (wider scatter)
- **Systematic bias:** LOESS curve deviates significantly from y=0, showing non-random residual pattern
- **Fan shape:** Variance slightly increases at extremes
- **High density:** Central region ($500K-$5M) has highest point density

**Interpretation:**
- Model exhibits **systematic conditional bias** dependent on price level
- **Underpredicts** mid-range properties ($250K-$500K)
- **Overpredicts** very low-price properties ($100K-$250K)
- **Variance increases** at extremes (heteroskedastic errors)

**Background:** White

**Assessment vs. Conference Standards:**
This is a **critical diagnostic for model calibration**:

**Strengths:**
1. ✅ **LOESS trend** reveals systematic bias
2. ✅ **Confidence band** shows variance pattern
3. ✅ **Standardized residuals** (divided by σ) makes scale interpretable
4. ✅ **Clear heteroskedasticity** visualization

**Weaknesses:**
1. ❌ **LOESS parameters not specified** (bandwidth, degree)
2. ❌ **Confidence band definition unclear** (±1σ? Pointwise CI?)
3. ❌ **No statistical test** for heteroskedasticity (e.g., Breusch-Pagan)
4. ❌ **No quantification** of bias magnitude
5. ❌ **X-axis scale ambiguous** (log or linear?)

**Critical Questions:**
1. Why does LOESS dip at $250K-$500K? Is this a data artifact or true model failure?
2. Is heteroskedasticity accounted for in uncertainty estimates?
3. Should the model use weighted loss or variance modeling?

**Recommended Enhancements:**
1. Explicitly label x-axis as log-scale: "Predicted Price (log scale)"
2. Add Breusch-Pagan test result: "Heteroskedasticity: BP = XXX, p < 0.001"
3. Specify LOESS parameters in caption
4. Add horizontal reference bands at ±1σ, ±2σ
5. Quantify max bias: "Max systematic bias: -0.5 log units at $350K (factor of 1.6×)"
6. Add text box explaining implications

**Conference Reviewer Likely Feedback:**
"Figure X shows clear heteroskedasticity and systematic conditional bias. The authors should discuss whether the Gaussian likelihood assumption is appropriate and consider variance modeling or robust loss functions."

**This figure is publication-critical and needs methodological discussion.**

---

### 4.11 residuals_vs_pred_bias.png

**Type:** Scatter plot with LOESS trend showing conditional bias

**Dimensions:** Wide format, approximately 8" × 5"

**Title:** "Conditional Bias"

**Axes:**
- **X-axis:** "Predicted Price\n(Values in Log Space)" with labels $100K to $250M
  - Appears to be in **original dollar units** despite subtitle
- **Y-axis:** "Residual" ranging from -1.0 to +1.0 (log units)

**Data Points:**
- Thousands of semi-transparent points (cyan/gray)
- High density in center, sparser at extremes

**Visual Elements:**
1. **Reference line (dashed black):** Horizontal at y = 0

2. **LOESS trend (solid red):** Shows dramatic bias pattern:
   - **$100K:** Residual ≈ +1.0 (model **overpredicts by factor of 2.7×**)
   - **$150K:** Sharp drop
   - **$250K:** Residual ≈ 0 (crossing zero)
   - **$250K-$1M:** Residual ≈ -0.1 (slight underprediction)
   - **$1M-$10M:** Residual ≈ -0.2 (moderate underprediction)
   - **$10M+:** Residual ≈ -0.1 (slight underprediction)

3. **Confidence band (shaded pink):** ±1 SD envelope
   - **Asymmetric and varying width:**
     - Very wide at $100K (±0.75)
     - Narrow at $500K-$2M (±0.3)
     - Widens again at high prices ($100M+: ±0.8)

**Key Finding:**
**Extreme bias at low prices:** The model systematically overpredicts properties under $200K by up to 2.7×, then becomes approximately unbiased for higher prices.

**Interpretation:**
- Model likely has **trouble with low-price properties**
- Possible causes:
  1. Different price-generating process for low-end properties
  2. Different building types dominate low-end (see residuals_by_bldg_class.png)
  3. Non-linear relationship not captured by model
  4. Floor effect (prices can't go below land value)

**Comparison to residuals_standardized_vs_pred.png:**
- **Same underlying data** but different y-axis range
- This version zooms in on [-1, +1] range, emphasizing low-price bias
- The other version shows [-2, +2] and includes more outliers

**Assessment vs. Conference Standards:**
This figure highlights a **critical model failure**:

**Strengths:**
1. ✅ **Dramatic bias clearly visible**
2. ✅ **Focus on interpretable range** (-1 to +1)
3. ✅ **Actionable insight** (model fails for low-price properties)

**Weaknesses:**
1. ❌ **Axis label inconsistency:** Says "Log Space" but uses dollar units
2. ❌ **No explanation** for extreme low-price bias
3. ❌ **No sample size** information (how many properties < $200K?)
4. ❌ **No proposed solution**

**Critical Questions:**
1. How many properties are in the $100K-$200K range where bias is extreme?
2. Are these predominantly Vacant Land / Religious buildings (see residuals_by_bldg_class.png)?
3. Should these be excluded or modeled separately?
4. Is this bias present in training data or only test set?

**Recommended Enhancements:**
1. Fix axis label: Remove "(Values in Log Space)" or change to log-scale axis
2. Add annotation: "Extreme overprediction for prices < $200K\n(factor of 2.7× at $100K)"
3. Add vertical line at $200K with label "Bias threshold"
4. Show sample size distribution (marginal histogram on x-axis)
5. Add inset showing breakdown by building class for $100K-$200K segment

**Conference Reviewer Likely Feedback:**
"The model shows severe bias for properties under $200K (overpredicting by 2.7×). The authors must explain this phenomenon and either fix the model or exclude these properties from evaluation. What fraction of the dataset is affected?"

**This figure reveals a critical flaw that must be addressed in the paper.**

---

## 5. Model Diagnostics

### 5.1 coverage_by_sale_year.png

**Type:** Line plot showing calibration over time

**Dimensions:** Wide format, approximately 10" × 5"

**Title:** "Uncertainty Calibration by Sale Year"

**Axes:**
- **X-axis:** "Sale Year" ranging from 2003 to 2022 (annual intervals)
- **Y-axis:** "Empirical Coverage" ranging from 0.0 to 1.0 (proportion scale)

**Reference Lines (horizontal dashed):**
1. **Top line (y = 0.95):** Gray dotted, labeled implicitly as 95% nominal coverage
2. **Middle line (y = 0.80):** Gray dotted, labeled implicitly as 80% nominal coverage
3. **Bottom line (y = 0.50):** Gray dotted, labeled implicitly as 50% nominal coverage

**Data Series (3 lines):**
1. **50% CI (blue circles):**
   - Starts at ~0.32 (2003) - **severely undercovered**
   - Jumps to ~0.60 (2004)
   - Stabilizes around 0.55-0.60 (2005-2022)
   - **Consistently undercovered** (should be at 0.50 line)
   - Slight downward trend over time

2. **80% CI (orange circles):**
   - Starts at ~0.53 (2003) - **severely undercovered**
   - Jumps to ~0.82 (2004)
   - Fluctuates around 0.75-0.80 (2005-2022)
   - **Slightly undercovered** (should be at 0.80 line)
   - Relatively stable

3. **95% CI (green circles):**
   - Starts at ~0.68 (2003) - **severely undercovered**
   - Jumps to ~0.90 (2004)
   - Stabilizes around 0.85-0.90 (2005-2022)
   - **Moderately undercovered** (should be at 0.95 line)
   - Most stable of the three

**Visual Patterns:**
- **2003 anomaly:** All three intervals show extreme undercoverage in 2003
  - Possible causes: Very small sample, market anomaly (post-9/11?), data quality issue
- **2004 improvement:** Sharp jump toward nominal coverage
- **2005-2022 stability:** Relatively consistent undercoverage across all intervals
- **Parallel tracking:** The three lines maintain roughly constant spacing

**Calibration Assessment:**
- **Miscalibrated:** Model consistently **underestimates uncertainty**
- **50% CI achieves ~57%** coverage (should be 50%) - **overconfident**
- **80% CI achieves ~76%** coverage (should be 80%) - **overconfident**
- **95% CI achieves ~88%** coverage (should be 95%) - **overconfident**

**Temporal Trend:**
- Slight downward drift over time suggests calibration worsening slightly
- No dramatic changes post-2008 (financial crisis) or post-2020 (COVID)

**Background:** White with light gray gridlines

**Legend:**
- Blue: "50% CI"
- Orange: "80% CI"
- Green: "95% CI"

**Assessment vs. Conference Standards:**
This is a **critical calibration diagnostic**:

**Strengths:**
1. ✅ **Multi-level coverage** (50%, 80%, 95%) provides comprehensive calibration check
2. ✅ **Temporal variation** reveals time-dependent miscalibration
3. ✅ **Clear reference lines** show nominal levels
4. ✅ **Long time span** (2003-2022) shows robustness

**Weaknesses:**
1. ❌ **No confidence bands** on empirical coverage (should show ±1.96√(p(1-p)/n) binomial CI)
2. ❌ **Sample sizes not shown** per year (critical for interpreting 2003)
3. ❌ **No statistical test** for miscalibration (e.g., binomial test, Hosmer-Lemeshow)
4. ❌ **No proposed solution** (recalibration? temperature scaling?)
5. ❌ **No explanation** for 2003 anomaly

**Critical Findings:**
1. **Systematic underestimation of uncertainty** across all confidence levels
2. **Calibration roughly constant over time** (2005-2022)
3. **2003 is an outlier** requiring investigation

**Recommended Enhancements:**
1. Add error bars: Binomial CIs for each point
2. Add sample size annotations: "n=500" below each year
3. Add text box with calibration metrics:
   ```
   Mean Absolute Coverage Error:
     50% CI: 0.07 (p < 0.001)
     80% CI: 0.04 (p < 0.001)
     95% CI: 0.07 (p < 0.001)
   ```
4. Add annotation explaining 2003: "2003: n=87, post-9/11 anomaly"
5. Discuss recalibration methods in caption

**Conference Reviewer Likely Feedback:**
"The model shows systematic miscalibration (undercoverage) across all confidence levels and years. The authors should apply temperature scaling or other recalibration methods and report post-calibration metrics. The 2003 anomaly requires explanation or exclusion."

**This figure is essential for publication but requires recalibration analysis.**

---

### 5.2 kurtosis_comparison.png

**Type:** Overlaid density curves comparing Normal vs. heavy-tailed distribution

**Dimensions:** Square aspect ratio, approximately 7" × 6"

**Title:** "Visualizing Kurtosis: Normal vs Heavy-Tailed"

**Axes:**
- **X-axis:** "Standard Deviations" ranging from -6 to +6 (units of σ)
- **Y-axis:** "Density" ranging from 0.0 to 0.5

**Plot Elements:**
1. **Normal distribution (dashed gray line):** Labeled "Normal (Kurt=0)"
   - Classic Gaussian bell curve
   - Peak at ~0.40 density
   - Tails approach zero by ±3σ

2. **Heavy-tailed distribution (solid blue line):** Labeled "Heavy Tails (Kurt approx 9.3)"
   - **Higher peak** at ~0.50 density (leptokurtic)
   - **Lower shoulders** at ±1σ to ±2σ
   - **Heavier tails** beyond ±2σ (visible mass at ±4σ, ±5σ)

3. **Shaded region (light blue):** Labeled "Excess Mass in Tails"
   - Shows area where heavy-tailed exceeds Normal in tail regions
   - Symmetric shading in both tails beyond approximately ±2σ

**Visual Comparison:**
- **Central peak (|x| < 0.5σ):**
  - Blue curve **taller** than gray (0.50 vs. 0.40)
  - More concentration at center

- **Shoulders (0.5σ < |x| < 2σ):**
  - Blue curve **lower** than gray
  - Less mass in "shoulders"

- **Tails (|x| > 2σ):**
  - Blue curve **higher** than gray
  - Shaded region shows excess mass
  - At x = ±4σ: Blue ≈ 0.01, Gray ≈ 0.0001 (100× more mass)

**Kurtosis Interpretation:**
- **Excess kurtosis ≈ 6.3** (total kurtosis 9.3 vs. 3.0 for Normal)
- Shape characteristic of Student-t with low degrees of freedom

**Background:** White with clean grid

**Legend (upper-right):**
- Dashed gray: "Normal (Kurt=0)"
- Solid blue: "Heavy Tails (Kurt approx 9.3)"
- Light blue shading: "Excess Mass in Tails"

**Assessment vs. Conference Standards:**
This is an **excellent pedagogical figure**:

**Strengths:**
1. ✅ **Clear visual demonstration** of kurtosis concept
2. ✅ **Direct comparison** of Normal vs. observed
3. ✅ **Shaded region** highlights excess tail mass
4. ✅ **Standardized x-axis** (in σ units) makes it generalizable
5. ✅ **Quantitative kurtosis** values in legend

**Weaknesses:**
1. ❌ **Heavy-tailed curve not labeled** with specific distribution (Student-t? Empirical?)
2. ❌ **No indication** this is the actual residual distribution vs. theoretical
3. ❌ **Kurtosis definition inconsistent:** Legend says "Kurt=0" for Normal (should be Kurt=3), "Kurt approx 9.3" for heavy-tailed
   - **Standard definition:** Normal has kurtosis = 3
   - **Excess kurtosis definition:** Normal has excess kurtosis = 0
   - **Appears to use excess kurtosis** in legend but not clearly stated

**Recommended Enhancements:**
1. Clarify kurtosis definition: "Normal (Excess Kurt=0, Kurt=3)"
2. Specify heavy-tailed distribution: "Fitted Student-t (ν=1.1, Excess Kurt≈6.3)"
3. Add caption: "Blue curve represents empirical residual distribution"
4. Annotate specific tail probabilities:
   ```
   P(|X| > 3σ):
     Normal: 0.27%
     Observed: 2.5% (9× higher)
   ```
5. Add vertical reference lines at ±2σ, ±3σ

**Minor Issue:**
The label "Kurt=0" for Normal is **technically incorrect** if using standard kurtosis definition. Should be:
- "Excess Kurtosis = 0" (recommended for clarity)
- OR "Kurtosis = 3" (standard definition)

**This figure is publication-ready with clarification of kurtosis definition.**

---

### 5.3 synthetic_missingness_stress.png

**Type:** Line plot showing model robustness to missing data

**Dimensions:** Approximately 6" × 4.5"

**Title:** "Missingness Test"

**Axes:**
- **X-axis:** "Fraction Masked" ranging from 0.0 to 0.5 (proportion of features artificially masked)
- **Y-axis:** Dual scale (two metrics plotted):
  - Left axis: Values 0-12 (appears to be error/loss metric)
  - Right axis: Not clearly labeled in description but likely corresponds to second metric

**Subtitle:** "(n = 500 samples. Values in Log Price Space.)"

**Data Series:**
1. **Root Mean Square Error (teal solid line with circle markers):**
   - Starts at ~1.2 (no masking)
   - Increases to ~5.5 (10% masked)
   - Increases to ~6.8 (20% masked)
   - Increases to ~8.0 (30% masked)
   - Increases to ~9.8 (50% masked)
   - **Roughly linear increase** with masking fraction

2. **Mean Predicted Sigma (purple dashed line with square markers):**
   - Starts at ~1.7 (no masking)
   - Increases to ~6.4 (10% masked)
   - Increases to ~8.0 (20% masked)
   - Peaks at ~11.0 (30% masked)
   - Decreases slightly to ~9.2 (50% masked)
   - **Non-monotonic** with peak at 30%

**Visual Patterns:**
- **Both metrics increase** with missingness, as expected
- **RMSE increases steadily** (roughly 0.16 per 1% masking)
- **Predicted uncertainty (sigma) increases faster** initially, then plateaus/decreases
- **Divergence at high masking:** Predicted sigma > RMSE at low masking, but RMSE catches up at high masking

**Interpretation:**
- **Model degrades gracefully:** Performance doesn't collapse with moderate missingness
- **Uncertainty awareness:** Predicted sigma increases, indicating model "knows" predictions are less reliable
- **Possible miscalibration at extremes:** Sigma decrease at 50% suggests model may be overconfident or hitting numerical issues
- **MIWAE architecture working:** Designed to handle missing data, and stress test confirms this

**Background:** White with gray gridlines

**Legend:**
- Teal solid with circles: "Root Mean Square Error"
- Purple dashed with squares: "Mean Predicted Sigma"

**Assessment vs. Conference Standards:**
This is a **valuable robustness diagnostic**:

**Strengths:**
1. ✅ **Systematic stress test** of key capability (handling missingness)
2. ✅ **Dual metrics** (error + uncertainty) show calibration
3. ✅ **Reasonable test range** (0-50% masking)

**Weaknesses:**
1. ❌ **Small sample size** (n=500) raises concerns about stability
2. ❌ **No error bars** (with n=500, there's significant sampling variability)
3. ❌ **No baseline comparison** (how does standard VAE or mean imputation perform?)
4. ❌ **Masking mechanism not specified** (MCAR? MAR? MNAR?)
5. ❌ **Single random seed?** Should average over multiple masks
6. ❌ **Non-monotonic sigma** at 50% not explained

**Critical Questions:**
1. **Why does sigma decrease at 50%?** Numerical instability? Insufficient samples? Model failure?
2. **Is masking MCAR?** (Missing Completely At Random) - this is crucial for interpretation
3. **How does this compare to the natural missingness in the dataset?**
4. **What's the masking scheme?** Random features? Random properties? Both?

**Recommended Enhancements:**
1. **Increase sample size** to n=5000+ or show error bands from bootstrap
2. **Add baseline comparisons:**
   - Standard VAE (without importance weighting)
   - Mean imputation
   - Listwise deletion
3. **Multiple random seeds:** Average over 10+ masks per fraction
4. **Add annotation:**
   ```
   Masking: MCAR, random features
   Error increase: ~0.16 per 1% masked
   Seeds: 10 random masks per point
   ```
5. **Investigate 50% anomaly** and add note if it's a real effect
6. **Add reference line:** Show natural missingness level in dataset (~25%?)

**Conference Reviewer Likely Feedback:**
"The missingness stress test with n=500 is too small to draw reliable conclusions. Please increase sample size, add error bars, and compare to baselines. The non-monotonic sigma at 50% masking requires explanation."

**This figure needs methodological strengthening before publication.**

---

## 6. Branding Assets

### 6.1 columbia_logo.png

**Type:** University logo (branding asset)

**Dimensions:** Approximately 300 × 100 pixels (wide format)

**Content:**
- Text: "**Columbia University**" (two lines)
- **Font:** Official Columbia University serif typeface
- **Color:** Columbia Blue (RGB: 0, 43, 127 / #002B7F)
- **Background:** White/transparent

**Usage:** Header element for poster and report

**Quality:** High resolution, suitable for print

---

### 6.2 columbia_engineering_logo.png

**Type:** School logo (branding asset)

**Dimensions:** Approximately 300 × 100 pixels (wide format)

**Content:**
- Text: "**Columbia\nEngineering**" (two lines)
- **Font:** Official Columbia University serif typeface
- **Color:** Columbia Blue (RGB: 0, 43, 127 / #002B7F)
- **Background:** White/transparent

**Usage:** Header element for poster

**Quality:** High resolution, suitable for print

---

## Summary Statistics

**Total Images:** 28

**By Category:**
- Model Architecture: 1
- Training Diagnostics: 2
- Latent Space Visualizations: 9 (6 publication-quality, 3 draft versions)
- Residual Analysis: 11 (7 publication-quality, 4 draft versions)
- Model Diagnostics: 3
- Branding: 2

**Publication Readiness:**
- ✅ **Ready for publication (with minor caption improvements):** 15
- ⚠️ **Needs enhancements:** 7
- ❌ **Do not use (draft/superseded):** 4
- 📋 **Branding assets:** 2

---

## Critical Assessment: Conference Standards Comparison

### Strengths vs. ICML/NeurIPS/AAAI Standards

**What the current figures do well:**
1. ✅ **Clear visual encodings** with appropriate colormaps
2. ✅ **Proper axis labels and units** in most cases
3. ✅ **Interpretable latent space** visualizations with feature attributions
4. ✅ **Comprehensive diagnostics** covering multiple aspects (residuals, calibration, spatial)
5. ✅ **Good use of reference lines** (Normal distributions, zero lines, etc.)
6. ✅ **Appropriate plot types** for each analysis

### Gaps vs. Conference Standards

**What needs improvement for top-tier publication:**

1. ❌ **Insufficient quantitative rigor:**
   - Missing statistical tests (spatial autocorrelation, calibration tests)
   - Missing confidence intervals/error bars
   - Missing sample sizes
   - Missing goodness-of-fit metrics

2. ❌ **Incomplete methodological transparency:**
   - LOESS parameters not specified
   - Hexbin size/parameters not specified
   - Color schemes for 100+ categories problematic
   - Fit methods not always clear (MLE? moment matching?)

3. ❌ **Missing baseline comparisons:**
   - No comparison to standard VAE, gradient boosting, etc. in most figures
   - Missingness test lacks baseline methods

4. ❌ **Weak uncertainty quantification:**
   - Calibration issues identified but no recalibration proposed
   - Error bars missing from most plots
   - Prediction intervals not evaluated systematically

5. ❌ **Unexplained anomalies:**
   - 2003 coverage anomaly in coverage_by_sale_year.png
   - 50% masking sigma decrease in synthetic_missingness_stress.png
   - Extreme low-price bias in residuals_vs_pred_bias.png
   - Non-monotonic patterns not discussed

### Comparison to Referenced Conferences

**ICML/NeurIPS expectations:**
- **Figures must be self-contained** ✅ Mostly achieved
- **Statistical significance required** ❌ Often missing
- **Baselines mandatory** ❌ Largely absent from figures
- **Reproducibility details needed** ⚠️ Partial
- **Limitations discussed** ⚠️ Visible but not discussed

**AAAI expectations:**
- **Applied work must show real-world impact** ✅ Spatial bias, building class analysis
- **Practical limitations addressed** ⚠️ Shown but not resolved
- **Clarity for interdisciplinary audience** ✅ Generally good
- **Actionable insights** ⚠️ Diagnostic but not prescriptive

### Overall Assessment

The figure collection demonstrates **solid scientific work** with **good visualization practices**, but falls short of **top-tier conference standards** in several areas:

**Current Level:** High-quality technical report / workshop paper
**Target Level:** ICML/NeurIPS main conference
**Gap:** Primarily methodological rigor and statistical testing

**Priority Improvements for Publication:**
1. **Add statistical tests and p-values** to all diagnostic figures
2. **Include baseline comparisons** (especially for missingness handling)
3. **Provide confidence intervals** on all point estimates
4. **Explain or address anomalies** (2003, 50% masking, low-price bias)
5. **Specify all methodological parameters** (LOESS bandwidth, hexbin size, etc.)
6. **Add sample size annotations** throughout
7. **Remove or aggregate** the 100+ category building class plot
8. **Discuss recalibration** and show post-calibration results

With these enhancements, the figures would meet ICML/NeurIPS standards.

---

## Recommendations for Report and Poster

### For the Report

**Main Text Figures (suggested):**
1. miwae_graphical_model.png (with enhanced caption)
2. convergence_loss.png (cleaner version)
3. latent_space_price.png (excellent interpretability)
4. latent_space_bldg.png (aggregated version, not original)
5. residuals_hist_references.png (shows Student-t fit)
6. miwae_residuals_qq.png (clear heavy tails)
7. residuals_spatial_map.png (WITH Moran's I in caption)
8. residuals_by_bldg_class.png (WITH sample sizes)
9. coverage_by_sale_year.png (WITH recalibration discussion)

**Supplementary Materials:**
- All remaining publication-quality figures
- Draft versions for showing development process (if relevant)

### For the Poster

**Essential Figures (space-limited):**
1. miwae_graphical_model.png (small, in Methodology)
2. convergence_loss.png (small, show training worked)
3. latent_space_price.png (large, main result)
4. latent_space_bldg.png (large, interpretability)
5. residuals_hist_references.png OR miwae_residuals_qq.png (one diagnostic, compact)
6. kurtosis_comparison.png (pedagogical, explains model choice)

### Figures to Avoid

**Never use in publication:**
- latent_space_bldg_orig.png (100+ colors, unreadable)
- miwae_latents_by_building_class.png (same issue)
- latent_space_price_decile_orig.png (superseded by better version)
- residuals_hist_simple.png (no reference distribution)
- residuals_qq_simple.png (superseded by cleaner version)

---

## Conclusion

This image set represents **extensive and thorough empirical analysis** of the Semi-Supervised MIWAE model. The visualizations effectively communicate key findings about latent space organization, heavy-tailed residuals, spatial bias, and calibration issues.

**Key Strengths:**
- Comprehensive coverage of model behavior
- Clear visual design and appropriate plot choices
- Interpretable latent space with feature attributions
- Honest assessment of model limitations

**Key Weaknesses:**
- Insufficient statistical rigor (missing tests, CIs, sample sizes)
- Some figures with too many categories (building classes)
- Missing baseline comparisons
- Unexplained anomalies in diagnostics
- Calibration issues identified but not resolved

**Path to Publication:**
With the recommended enhancements (primarily adding statistical tests, confidence intervals, sample sizes, and baseline comparisons), this work would meet the standards for publication at top-tier ML conferences (ICML, NeurIPS, AAAI).

**Estimated Effort:**
- **Minor caption updates:** 4-6 hours
- **Statistical test additions:** 8-12 hours of analysis + 4 hours figure updates
- **Baseline comparisons:** 16-24 hours of computation + 6 hours figure creation
- **Recalibration experiments:** 12-16 hours

**Total:** ~50-70 hours to bring figures to top-tier conference standards.

---

**Document End**
