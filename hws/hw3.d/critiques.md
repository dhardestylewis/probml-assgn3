# Document Critiques and Revision Plan

> [!CAUTION]
> **Backup branch**: `backup-edits` – push regularly with `git push origin backup-edits`

**Priority**: P1=Critical/Visible, P2=Important, P3=Polish

---

## Section 1: Abstract (Lines 23, 25)

### Completed [x]
- [x] `will develop` → `develops`
- [x] `I will treat` → `I treat`
- [x] `observed log sale prices` → `observations of sale prices`
- [x] `I plan to implement` → `I implement`
- [x] `MIWAE-style variational autoencoder` → `variant of the Missing-data Importance-Weighted Autoencoder (MIWAE)`
- [x] `over the latent space` → `over the latent variables`
- [x] `The model will operate` → `The model operates`
- [x] `perform` → `performs`, `produce` → `produces`
- [x] `latent codes` → `latent variables`
- [x] `intermittently missing` → `mostly missing`
- [x] Removed `operates on masked inputs`
- [x] `I will compare` → `I compare`
- [x] Simplified scores to `held-out predictive log-likelihood, RMSE, MAE, MAPE`
- [x] `I will also assess` → `I also assess`
- [x] `I will visualize` → `I visualize`
- [x] Removed parenthetical latent space description
- [x] Rewrote implications with 'why'

### Outstanding [USER]
- [x] **P1** "heavy-tailed noise" – ✓ Added kurtosis 9.27 to line 175
- [ ] **P2** "large panel of roughly $10^5$ parcels" – does size matter or just order of magnitude? Clarify
- [ ] **P2** Are we treating observations as noisy in code? Gaussian likelihood implies noise – state explicitly
- [x] **P2** "mostly missing at 93%" – ✓ VERIFIED: 25% missing (75% observed) per notebook log
- [x] **P2** "observed log sale prices" → "observations of sale prices" (no log in abstract) – applied
- [ ] **P2** Is it "MIWAE-style" or just "variant of MIWAE"? Look up MIWAE paper to verify
- [ ] **P2** "semi-supervised MIWAE-style variational autoencoder" – repetitive? Many keywords include others. Separate clauses
- [ ] **P2** "latent space" – is this Blei vocabulary? Check 2025f readings, create reference doc
- [ ] **P2** "latent codes" – within community vocabulary? Recast using existing definitions
- [ ] **P2** "discriminative baseline" – what does this mean vs just "baseline"?
- [ ] **P2** "(e.g., gradient-boosted trees on same features)" – remove parenthetical, make proper clause
- [ ] **P2** "log-posterior-predictive scores" – which specific scores? Name them. But no log-space detail in abstract
- [ ] **P2** Convert all "will"s to present tense? Review hw3 to decide
- [ ] **P2** "standard supervised approaches" – cite these later?
- [ ] **P2** "noisy, heavy-tailed real-estate data" – cite other evidence for this claim
- [ ] **P2** "significant implications for fair property taxation... critical for equitable policy" – why? State the because/why immediately after
- [x] **P2** **Citation**: "large panel of NYC property records" → cite MapPLUTO and ACRIS – added [2, 3]
- [ ] **P2** **Specificity**: "disproportionately affect specific neighborhoods" → state *what* effects (higher tax burden, undervaluation)
- [x] **P2** **Citation**: "Accurate, uncertainty-aware valuations are critical..." → cite source – added [5] (Berry & Bednarz)
- [ ] **P3** **Abstract Metrics**: Replace "MAPE" with "Median Absolute Error" to align with findings.
- [ ] **P3** **Consistency**: Standardize parcel count: use rounded (e.g., "approx 120k") in Abstract/Intro, and exact (e.g., "122,712") in Data section.

### Outstanding [SELF]
- [ ] **P3** Line 23: "feature distribution" → "covariate distribution"
- [ ] **P3** Line 23: "extreme residuals" – undefined at this point, move explanation or rephrase
- [ ] **P3** Line 23: **Sentence length** – 5+ clauses, split into 2-3 sentences
- [ ] **P3** Line 25: "held-out predictive log-likelihood" – define for non-ML readers?
- [ ] **P3** Line 25: "systematic gaps in market coverage" – explain or remove
- [ ] **P3** Line 25: "broader aim" – weak transition
- [ ] **P3** Line 25: **Scope creep** – "significant implications for fair property taxation" appropriate for HW?

---

## Section 2: Introduction (Lines 31, 33, 35)

### Completed [x]
- [x] Removed "In this section"
- [x] Rewrote numbered goals as prose
- [x] `meaningful structure` → `spatial and physical heterogeneity`
- [x] `(log) sale price` → `log-transformed sale prices`
- [x] Line 35: `flexible generative model` → `latent variable model with non-linear likelihoods`
- [x] Line 35: `natural` → `appropriate`
- [x] Line 35: Verb conjugation: `models`, `provides`
- [x] Line 35: Removed `end-to-end`
- [x] Line 35: `latent factors` → `latent variables`

### Outstanding [USER]
- [ ] **P2** **Vocabulary**: "latent representation" – is this Blei vocabulary? Check 2025f readings
- [x] **P2** **Citation**: Line 33 "Accurate property valuation is fundamental..." → cite IAAO (2013) – added [4]
- [x] **P2** **Citation**: Line 33 "systematic errors... disproportionately affect" → cite Berry & Bednarz (1975) – added [5]
- [ ] **P2** **Substantiation**: "noisy" and "heavily skewed" – show evidence before claiming (kurtosis 9.27, skew figure)

### Outstanding [SELF]
- [ ] **P3** Line 31: "semi-supervised, probabilistic latent factor model" – too many adjectives, streamline
- [ ] **P3** Line 31: "physical heterogeneity" – define or example (building age, lot size)
- [ ] **P3** Line 33: "traditional automated valuation models (AVMs)" – cite examples or remove acronym
- [ ] **P3** Line 33: "By explicitly modeling these factors" – antecedent unclear
- [x] **P3** Line 33: "we aim" – inconsistent pronoun (rest is "I") – fixed
- [ ] **P3** Line 35: "multivariate" – correct term? vs "high-dimensional"
- [ ] **P3** Line 35: "non-linear likelihoods" – VAE uses linear Gaussian decoder, clarify
- [ ] **P3** Line 35: "exploits shared structure" – vague, structure across what?

---

## Section 3: Model (Lines 37-79)

### Completed [x]
- [x] `Prior on latent factors` → `Prior on latent variables`
- [x] `but conceptually it is` → `This parameterization induces`
- [x] `(log) sale price` → `log-transformed sale price`
- [x] `price head` → `prediction network`
- [x] `schematic graphical model` → `graphical model`

### Outstanding [USER]
- [x] **P2** **Parenthetical**: Line 39 "(parcel and building characteristics)" → rewrite as "...describing parcel and building characteristics" – fixed
- [ ] **P2** **Contextualization**: Line 46 "unconstrained logits and Cholesky factors" → explain what these mean
- [ ] **P2** **Repetition**: Line 46 "mixture of heavy-tailed components" → state once with equations, then back-reference

### Outstanding [SELF]
- [x] **P3** Line 39: "For each property we observe" – passive – fixed
- [ ] **P3** Line 39: "for a subset" – vague, which subset?
- [ ] **P3** Line 39: "$K = 3$" – why 3? justify
- [ ] **P3** Line 41: "I place a" – unnecessary, just "The prior is..."
- [ ] **P3** Line 46: "multi-modal, robust structure" – "robust" to what?
- [ ] **P3** Line 46: "latent space" → "latent variables"
- [ ] **P3** Line 48: **Audience**: "diagonal-covariance Gaussian decoder" – define for non-experts
- [ ] **P3** Line 52: "In practice" – remove weak phrasing
- [ ] **P3** Line 52: "approximately zero mean" → "standardized"
- [ ] **P3** Line 58: "small prediction network" – specify architecture (two-layer MLP, widths 8,4)
- [ ] **P3** Line 58: **Audience**: "conditional generative model" – define
- [ ] **P3** Line 60: "full joint distribution" – redundant "full"
- [ ] **P3** Line 70: "Missing prices simply remove" – passive/awkward
- [ ] **P3** Line 72: Figure reference before figure – move after

---

## Section 4: Inference (Lines 80-113)

### Completed [x]
- [x] Line 82: "mixture-structured" → "multimodal"
- [x] Line 88: "implemented by a neural network" → "parameterized by a neural network"
- [x] Line 104: Parenthetical "(mixture prior...)" → colon list
- [x] Line 104: "price head" → "prediction network"
- [x] Line 104: "scalar hyperparameter" → "hyperparameter"
- [x] Line 104: "upweights" → "emphasizes"
- [x] Line 104: "latent factors" → "latent variables"

### Outstanding [USER]
*(Add as you review)*

### Outstanding [SELF]
- [ ] **P3** Line 82: "intractable because" – citation needed?
- [ ] **P3** Line 82: **Audience**: "amortized variational inference" – define or cite
- [ ] **P3** Line 84: "For each property I introduce" – wordy
- [ ] **P3** Line 88: "marginalized out analytically" – how? explain
- [ ] **P3** Line 90: **Audience**: "importance samples" – define on first use
- [ ] **P3** Line 90: "partially observed $(x_i, y_i)$" – clarify: partial in $x$ or $y$?
- [ ] **P3** Line 90: **Notation**: "$K_{\text{IW}}$" – define, confusing with $K$ latent dim
- [ ] **P3** **Notation consistency**: Use distinct letters: $K$ (latent), $M$ (mixture), $S$ (samples)
- [x] **P1** Line 106: ✓ Filled epoch count (50 epochs)
- [ ] **P3** Line 106: "early stopping" – describe criterion (patience?)
- [x] **P2** Line 111: ✓ Added MIWAE citation [1] in References section
- [ ] **P3** Line 111: Figure caption "SemiSupMIWAE" – define acronym

---

## Section 5: Data and Setup (Lines 115-139)

### Completed [x]
- [x] Line 127: "MIWAE masking mechanism" → "zero-masking strategy"
- [x] Line 127: Clarified "receive binary masks" and "reconstruction loss"
- [x] Line 134: "Price head" → "Prediction network"

### Outstanding [USER]
- [ ] **P2** **Justification**: "How does this setup align with your project goals?" – Explain *why* the 80/20 random split is appropriate (e.g., testing interpolation within the same spatial domain vs. extrapolation).

### Outstanding [SELF]
- [ ] **P3** Line 117: "after filtering and preprocessing" – what filtering?
- [ ] **P3** Line 117: "parcel-level record" – redundant with line 31
- [x] **P2** Line 117: ✓ Added MapPLUTO [2], ACRIS [3] in References section
- [ ] **P3** Line 119-122: "etc." – lazy, list all or "including but not limited to"
- [x] **P1** Line 125: ✓ VERIFIED: 75% observed is correct per notebook (25% = 30,678/122,712 missing)
- [ ] **P3** Line 125: "used only through their $x_i$" → "contribute only to unsupervised objective"
- [ ] **P3** Line 129-136: Missing batch size, learning rate, optimizer
- [ ] **P3** Line 132: "$\nu = 4$" – justify choice
- [ ] **P3** Line 133: "width 21" – why 21?
- [ ] **P3** Line 138: "cross-validation predictions unavailable" – why?
- [ ] **P3** **Reproducibility**: Missing random seed, code repository, package versions

---

## Section 6: Results (Lines 140-213)

### Completed [x]
- [x] Line 142: Removed meta-commentary "I report both..."
- [x] Line 142: "latent factors" → "latent variables"
- [x] Line 152: "numerically dominated" → "skewed by"
- [x] Line 185: "price head" → "prediction network" (2 occurrences)
- [x] Line 185: Removed "(i)...and (ii)" parenthetical numbering
- [x] Line 189: "leaves roughly half...on the table" → "fails to capture approximately 50%"
- [x] Line 189: "price head" → "prediction network"
- [x] Line 193: "latent factors" → "latent variables"

### Outstanding [USER]
*(Add as you review)*

### Outstanding [SELF]
- [ ] **P3** Line 146: "approximate Gaussian posterior" – in $z$ or $y$?
- [ ] **P3** Line 149-152: "$\approx$" – give exact values
- [ ] **P3** Line 152: "not representative" – remove or justify
- [x] **P1** Line 167-168: ✓ Filled median ~0.25, 95th pct ~4.1 log units
- [x] **P2** Table 1: Remove Notes column – applied
- [x] **P2** Table 1: Report only in $ (not log) – applied
- [x] **P2** Table 1: RMSE label → "RMSE (price in $)" – applied
- [x] **P2** Table 1: Remove "random 20% holdout", "robust central error", "tail error" – applied
- [x] **P2** Table 1: Remove ~ (approx) symbols – applied
- [x] **P2** Table caption: mention "posterior mean ŷ" once – applied
- [ ] **P3** Line 171: Table caption too long
- [ ] **P3** Line 175: "on the order of" – exact values
- [ ] **P3** Line 175: **Sentence length** – mixes three ideas, split
- [ ] **P2** Line 175: "substantial kurtosis" → give number (9.27)
- [ ] **P3** **Paragraph unity**: Lines 175-176 mixes median error, percentiles, kurtosis
- [ ] **P3** Line 185: **Claim strength**: "preserve most of the predictive signal" → quantify
- [ ] **P3** Line 189: "model misspecification" – explain why
- [ ] **P3** Line 198: "value/size continuum" → "correlates with assessed value"
- [ ] **P3** Line 202: "weaker but visible" – subjective, quantify
- [ ] **P3** Line 202: **Parenthetical**: "(e.g., moderate $R^2$...)" → main text
- [ ] **P3** Line 204: **Parenthetical**: "(e.g., small 1--4 family...)" → clause
- [ ] **P3** Line 204: "latent factors" → "latent variables"
- [ ] **P3** Line 209: Figure caption "latent factors" → "latent variables"
- [ ] **P3** Line 213: "compact latent representation" – vary language
- [ ] **P3** Line 213: "leftover signal" → "residual variance"
- [ ] **P3** Line 213: **Parenthetical**: "(e.g., heavier-tailed...)" → Plans section
- [ ] **P3** Line 213: **Claim strength**: "competitive predictive performance" → vs what?

---

## Section 7: Plans and Future Work (Lines 215-231)

### Completed [x]
*(None yet)*

### Outstanding [USER]
*(Add as you review)*

### Outstanding [SELF]
- [ ] **P3** Line 217: "Plans for Final Weeks" → "Planned Extensions"
- [ ] **P3** Line 218: "In the remaining weeks, I plan to:" – meta, remove
- [ ] **P3** Line 220: "price head" → "prediction network"
- [ ] **P3** Line 220: **Parenthetical**: "(e.g., Student-t)" → integrate
- [ ] **P3** Line 222: "richer architectures" → specify: ResNets, attention, deeper MLPs
- [ ] **P3** Line 226: "Current Bottlenecks" → "Limitations"
- [ ] **P3** Line 227: "unused signal" – symptom not cause, clarify bottleneck
- [ ] **P3** Line 227: "suggests potential model misspecification or optimization" – hedging, pick one
- [ ] **P3** Line 227: "limits the number of particles" – give actual number
- [ ] **P3** Line 230: **Parenthetical**: "(e.g., via a graph neural network...)" → integrate
- [ ] **P3** Line 230: "exploring fully unsupervised pre-training" – passive

---

## Blei Vocabulary Reference (TO BUILD)
Track inclusion/exclusion of terms from 2025f readings:
- [ ] "latent space" – check
- [ ] "latent codes" – check  
- [ ] "latent representation" – check
- [ ] "latent variables" – likely standard
- [ ] "latent factors" – check
