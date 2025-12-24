# Referenced Publications Analysis
**Created:** 2025-12-23
**Purpose:** Comparative analysis of referenced publications for STCS6701 Final Project

---

## 1. MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets

**Citation:** Mattei, P.-A. and Frellsen, J. (2019). MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97:4413--4423.

**URLs:**
- Official Proceedings: https://proceedings.mlr.press/v97/mattei19a.html
- arXiv preprint: https://arxiv.org/abs/1812.02633
- GitHub Implementation: https://github.com/pamattei/miwae

### Content & Focus
- Addresses missing-at-random (MAR) data in deep latent variable models
- Presents importance-weighted autoencoder approach for handling incomplete datasets
- Focuses on maximizing tight lower bound of log-likelihood for observed data
- Provides single imputation and generative modeling capabilities

### Methods & Equations
- **Core Method:** Importance-weighted ELBO (Evidence Lower Bound)
- **Key Equation:** Uses importance sampling to handle missing data patterns
- Variational inference with amortized encoders
- Monte Carlo estimation with multiple importance samples

### Metrics
- Log-likelihood on observed data
- Imputation accuracy measured by RMSE
- Comparative analysis against MICE, MissForest, GAIN

### Depth & Detail
- **Length:** 11 pages (ICML format)
- **Depth:** Detailed mathematical derivations, proofs in appendix
- **Experiments:** Multiple continuous datasets (MNIST, UCI repositories)
- **Organization:** Introduction → Method → Experiments → Conclusion

### Comparison to Our Work
**Similarities:**
- Both use importance-weighted variational inference
- Both handle missing data in latent variable models
- Semi-supervised extension is natural next step

**Differences:**
- MIWAE focuses on unsupervised imputation; our work adds supervised head for price prediction
- MIWAE uses standard Gaussian prior; we use Student-t mixture for heavy tails
- Our work targets specific application domain (real estate) vs. general methodology

**Our Contribution:**
- Semi-supervised architecture for joint generative and predictive modeling
- Heavy-tailed prior to match domain characteristics
- Application to NYC property valuation with policy implications

---

## 2. IAAO Standard on Ratio Studies (2013)

**Citation:** International Association of Assessing Officers (IAAO). (2013). *Standard on Ratio Studies*. Kansas City, MO.

**URL:** https://www.iaao.org/wp-content/uploads/Standard_on_Ratio_Studies.pdf

### Content & Focus
- Industry standard for property assessment quality
- Establishes benchmarks for assessment level, uniformity, and equity
- Focuses on ratio statistics and diagnostic metrics

### Methods & Metrics
- **Median Ratio:** Target range 0.90 to 1.10
- **COD (Coefficient of Dispersion):** Measures horizontal equity
- **PRD (Price-Related Differential):** Target 0.98 to 1.03
  - PRD < 0.98 indicates progressivity (higher prices → higher ratios)
  - PRD > 1.03 indicates regressivity (higher prices → lower ratios)
- **PRB (Price-Related Bias):** Target -0.05 to +0.05

### Depth & Detail
- **Length:** Technical standard document (30+ pages)
- **Depth:** Operational guidelines, formulas, interpretation
- **Organization:** Standards → Rationale → Examples → Appendices

### Comparison to Our Work
**Relevance:**
- Provides assessment quality benchmarks our model should meet
- PRD and PRB metrics directly apply to our residual analysis
- Horizontal and vertical equity concepts inform fairness evaluation

**Gaps in Our Work:**
- We report RMSE, MAE, log-likelihood but not IAAO-standard PRD/PRB
- Could add ratio study diagnostics to Results section
- Should discuss equity implications using IAAO framework

**Future Work:**
- Compute PRD, PRB, COD on held-out predictions
- Compare model equity to jurisdiction standards
- Frame policy implications using IAAO terminology

---

## 3. Berry & Bednarz (1975) Hedonic Model

**Citation:** Berry, B. J. L. and Bednarz, R. S. (1975). A Hedonic Model of Prices and Assessments for Single-Family Homes. *Land Economics*, 51(1), 21--40.

**URL:** RePEc handle: `RePEc:uwp:landec:v:51:y:1975:i:1:p:21-40`

### Content & Focus
- Early application of hedonic regression to property valuation
- Investigates relationship between assessments and market prices
- Questions: "Does assessor follow market or market follow assessor?"

### Methods & Equations
- **Core Method:** Hedonic regression (OLS)
- **Variables:** Property characteristics (size, age, location)
- **Time Series:** 7-year monthly sales data (Chicago bungalows)

### Metrics
- Standard OLS diagnostics (R², residuals)
- Assessment-to-price ratios
- Geographic and temporal variation analysis

### Depth & Detail
- **Length:** 20 pages (journal article)
- **Depth:** Empirical analysis, limited theoretical framework
- **Organization:** Literature Review → Data → Model → Results → Discussion

### Comparison to Our Work
**Similarities:**
- Both model property prices using covariates
- Both examine assessment equity (implicit vs. explicit)
- Both use real transaction data

**Differences:**
- Berry uses linear hedonic regression; we use deep latent variable model
- Berry focuses on assessor behavior; we focus on prediction and uncertainty
- Berry's era (1970s) had limited computational tools; we leverage modern ML

**Our Contribution:**
- Probabilistic uncertainty quantification (not available in OLS)
- Handles missing data explicitly (Berry requires complete cases)
- Joint generative model learns latent structure

---

## 4. Recent Urban Computing Literature (2021-2023)

### Key Themes from Search Results

#### Machine Learning for Property Valuation
**Sources:**
- "AI-Based on Machine Learning Methods for Urban Real Estate Prediction: A Systematic Survey" (Springer, 2023)
- "Boosting the accuracy of property valuation with ensemble learning and explainable artificial intelligence" (Annals of Regional Science, 2025)
- "Automated real estate valuation with machine learning models using property descriptions" (ScienceDirect, 2022)

**Findings:**
- ML models outperform traditional hedonic models by 18.4% (MAE reduction)
- Gradient boosting and ensemble methods achieve highest accuracy
- Feature importance: age, geographic coordinates, transit accessibility
- Urban amenity density and diversity positively correlate with price

**Comparison to Our Work:**
- We use probabilistic model; they use discriminative ML (GBM, XGBoost)
- We focus on uncertainty; they focus on point prediction accuracy
- We handle missing data; they typically require complete cases

**Gap:**
- Most ML papers lack uncertainty quantification
- Limited treatment of assessment equity vs. prediction accuracy
- Our work bridges probabilistic modeling and applied urban policy

---

## 5. Fairness in Property Taxation (2021-2023)

### Key Studies
**Sources:**
- "Assessment Frequency and Equity of the Property Tax" (Philadelphia Fed WP 21-43, 2021)
- "Reimagining property tax: AI-powered assessment" (WJAETS, 2024)
- "A Study of Equity in NC Property Tax Appraisals" (UNC Coates' Canons, 2023)

**Findings:**
- **Regressivity is widespread:** Lower-priced homes pay higher effective tax rates in nearly every US community
- **NC Study (2022):** 81 of 100 counties failed vertical equity standards (PRD outside 0.98-1.03)
- **Philadelphia (2019 reassessment):** Frequent reassessment improves horizontal equity, but gains smaller in disadvantaged areas
- **Infrequent reappraisal:** Causes assessment disparities, especially in rapidly appreciating areas

**Implications for Our Work:**
- **Motivation:** Our model addresses systematic errors that create inequity
- **Metrics:** Should report PRD/PRB to demonstrate equity performance
- **Policy Context:** Uncertainty-aware valuations help identify neighborhoods at risk of regressive assessment
- **Future Work:** Spatial disaggregation of model performance by income/race

---

## 6. Uncertainty Quantification in AVMs (2021-2025)

### Key Papers
**Sources:**
- "Towards a Better Uncertainty Quantification in Automated Valuation Models" (J. Real Estate Finance & Economics, 2024)
- "Uncertainty quantification in automated valuation models with spatially weighted conformal prediction" (Int'l J. Data Science Analytics, 2025)
- "Towards a Better Uncertainty Quantification in Automated Valuation Models" (arXiv:2312.06531, 2023)

**Methods:**
- **Conformal Prediction:** Model-agnostic confidence sets for ML predictions
- **Spatial Weighting:** Accounts for geographic autocorrelation in house prices
- **Ensemble Methods:** Bootstrap, quantile regression, stacked generalization
- **Bayesian Approaches:** Posterior predictive distributions

**Findings:**
- Nonlinear ML models (RF, GBM) lack native uncertainty estimates
- Direct application of standard UQ methods fails under spatial dependence
- Conformal prediction provides calibrated coverage without distributional assumptions
- Stakeholders need uncertainty estimates for risk-aware decision-making

**Comparison to Our Work:**
- **Our Approach:** Probabilistic model with native uncertainty (posterior predictive)
- **Their Approach:** Post-hoc UQ for black-box ML models
- **Advantage of Our Method:** Principled Bayesian uncertainty, interpretable latent structure
- **Advantage of Theirs:** Model-agnostic, works with best-performing discriminative models

**Synthesis:**
- Hybrid approach possible: Use our latent variables as input to conformal-wrapped GBM
- Could report both posterior predictive intervals and conformal prediction sets
- Spatial CP methods address autocorrelation we currently ignore

---

## Comparative Summary Table

| Aspect | MIWAE (2019) | IAAO (2013) | Berry (1975) | Urban ML (2021-23) | UQ in AVMs (2021-25) | Our Work |
|--------|-------------|-------------|--------------|-------------------|---------------------|----------|
| **Domain** | General ML | Assessment Standards | Urban Economics | Real Estate ML | AVM Methodology | NYC Property Valuation |
| **Method** | Importance-weighted VAE | Ratio Statistics | Hedonic Regression | Ensemble ML | Conformal Prediction, Bayes | Semi-supervised MIWAE + Student-t |
| **Primary Goal** | Missing data imputation | Quality benchmarks | Market-assessment relation | Prediction accuracy | Uncertainty quantification | Joint prediction & UQ |
| **Uncertainty** | Implicitly via ELBO | Not addressed | OLS standard errors | Rarely addressed | Central focus | Posterior predictive distribution |
| **Missing Data** | Core focus (MAR) | Not addressed | Excluded (listwise deletion) | Typically excluded | Not primary focus | Explicit missingness model |
| **Equity** | Not addressed | Central focus (PRD, PRB, COD) | Implicit (residuals) | Rarely addressed | Not primary focus | Discussed, not quantified |
| **Heavy Tails** | Gaussian assumptions | Outlier diagnostics | Normal residuals | Robust losses (Huber, quantile) | Distribution-free methods | Student-t mixture prior |
| **Length** | 11 pages (ICML) | 30+ pages (technical standard) | 20 pages (journal) | 15-25 pages (typical) | 15-30 pages (typical) | 10 pages (course report) |
| **Organization** | Intro → Method → Experiments → Conclusion | Standards → Rationale → Examples | Literature → Model → Results | Survey or empirical study | Method → Application → Results | Intro → Model → Inference → Results |

---

## Gaps & Opportunities for Final Report

### Critical Additions
1. **IAAO Metrics:** Compute PRD, PRB, COD and compare to standards (0.98-1.03 for PRD)
2. **Equity Analysis:** Report metrics by price quartile and building class to assess vertical equity
3. **Spatial Analysis:** Acknowledge spatial autocorrelation (currently ignored), compare to spatial CP methods
4. **Heavy-tail References:** Cite recent work on robust losses or Student-t likelihoods in AVMs

### Citations to Add
1. Recent ML survey papers (Springer 2023 systematic survey)
2. Philadelphia Fed working paper on assessment equity
3. Conformal prediction paper for AVMs (2024-2025)
4. Urban amenity/accessibility papers if discussing latent interpretations

### Style Alignment
- **ICML Style:** Our report is 10 pages, informal. ICML papers: 8-10 pages, two-column, dense
- **IAAO Style:** Technical, prescriptive, example-driven
- **Economics Style (Land Econ):** Literature review, theoretical framework, empirical results, policy discussion
- **ML Conference Style:** Concise abstract, method, experiments, ablations, discussion

**Recommendation for Final Report:**
- Adopt ICML two-column format (as specified in task)
- Add IAAO-standard metrics to Results section
- Expand References to include 2021-2025 urban ML and UQ papers
- Tighten notation and definitions to match ICML standards

---

## Document Metadata
- **Last Updated:** 2025-12-23
- **Sources:** Web searches, official proceedings, institutional standards
- **Verification:** All URLs and citations checked December 2025
