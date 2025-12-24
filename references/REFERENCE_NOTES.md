# Reference Papers Notes

**Date:** 2025-12-23
**Status:** Consolidated from multiple sources.

## Part 1: Recent Downloads & Annotations

### 1. MIWAE (2019)
**File:** `2019_Mattei_MIWAE.pdf` (was `01_miwae_2019.pdf`)
**Citation:** Mattei & Frellsen, ICML 2019.
**Key Relevance:** The core methodology for the project.
**Notes:** 
- Uses Importance-Weighted Autoencoder (IWAE) bound.
- Handles MAR data.
- Project extension: Semi-supervised, Student-t prior.

### 2. IAAO Standard on Ratio Studies (2013)
**File:** `2013_IAAO_RatioStudies.pdf` (was `02_iaao_2013.pdf`)
**Citation:** IAAO, 2013.
**Key Relevance:** Evaluation metrics for property assessment.
**Notes:** 
- Defines PRD (Price-Related Differential), PRB (Price-Related Bias), COD (Coefficient of Dispersion).
- Standard for assessment equity.

### 3. AI-Based ML for Urban Real Estate (2023)
**File:** `2023_Zhou_RealEstateMLSurvey.pdf` (was `04_zhou_2023_systematic_survey.pdf`)
**Citation:** Zhou et al., 2023.
**Key Relevance:** State-of-the-art context.
**Notes:** 
- Surveys ML methods in real estate.
- Supports the claim that most methods lack uncertainty quantification.

### 4. Assessment Frequency and Equity (2021)
**File:** `2021_Coombes_AssessmentEquity.pdf` (was `08_coombes_2021_phila_fed.pdf`)
**Citation:** Coombes et al., Philadelphia Fed.
**Key Relevance:** Fairness motivation.
**Notes:** 
- Discusses how infrequent assessment leads to inequity.
- Motivation for automated/frequent valuation models.

### 5. AI-Powered Assessment (2024)
**File:** `2024_WJAETS_AIPropertyTax.pdf` (was `09_ai_property_tax_2024.pdf`)
**Key Relevance:** Recent application.
**Notes:** 
- "Reimagining property tax" with AI.

### 6. Conformal Prediction in AVMs (2023/2025)
**File:** `2023_Wilms_ConformalPredictionAVM.pdf` (was `13_conformal_prediction_2023.pdf`)
**Citation:** Wilms et al., arXiv:2312.06531
**Key Relevance:** Uncertainty Quantification competitor.
**Notes:** 
- Uses conformal prediction for intervals.
- Good comparison point for Bayesian approach.

---

## Part 2: Core Methodology & Formatting Analysis

### Core Methodology Papers

| Ref | Paper | New Filename | Pages | Sections | Notes |
|-----|-------|--------------|-------|----------|-------|
| [1] | VAE (Kingma 2014) | `2014_Kingma_VAE.pdf` | 14 | Intro, Method, Related, Experiments, Conclusion | Standard ICLR format |
| [2] | IWAE (Burda 2016) | `2016_Burda_IWAE.pdf` | 14 | Intro, Background, IWAE, Experiments, Conclusion | Standard ICLR format |
| [3] | MIWAE (Mattei 2019) | `2019_Mattei_MIWAE.pdf` | 11 | Intro, Background, MIWAE, Experiments, Conclusion | Standard ICML format |
| [4] | Semi-supervised VAE (Kingma 2014) | `2014_Kingma_SemiSupervisedVAE.pdf` | 9 | Intro, Method, Experiments, Conclusion | NeurIPS format |

### Statistical Foundations

| Ref | Paper | New Filename | Notes |
|-----|-------|--------------|-------|
| [5] | Student-t Modeling (Lange 1989) | N/A | JASA - journal article, not conference |
| [6] | Calibration (Gneiting 2007) | `2007_Gneiting_ProbabilisticForecasting.pdf` | JRSSB - journal article |
| [7] | Missing Data (Little & Rubin 2019) | N/A | Textbook (not downloaded) |

### Property Valuation Literature

| Ref | Paper | Notes |
|-----|-------|-------|
| [8] | Hedonic Pricing (Rosen 1974) | JPE - foundational hedonic paper |
| [9] | Hedonic Assessments (Berry 1975) | Land Economics - property assessment |
| [10] | IAAO Standards (2013) | Industry standard document (`2013_IAAO_RatioStudies.pdf`) |

## Typical ICML/NeurIPS Section Lengths

Based on downloaded papers:

- **Abstract**: 150-200 words
- **Introduction**: 1-1.5 pages
- **Background/Related Work**: 0.5-1 page
- **Method/Model**: 2-3 pages
- **Experiments/Results**: 2-3 pages
- **Conclusion**: 0.5 page
- **References**: 0.5-1 page (15-30 citations typical)
