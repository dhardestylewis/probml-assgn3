# HW3 Project TODO

## Priority Tasks

### P1 - Critical Completeness
- [ ] Review new GUIDELINES.md Section 0 (Version Control Safety) [Added: 2025-12-11 23:16]
- [x] Add `(UNREVIEWED) [Added: timestamp]` tags to GUIDELINES.md Sections 1-7 and all subsections [Added: 2025-12-11 23:21] [Completed: 2025-12-11 23:27]

### P1 - Poster Finalization (UNREVIEWED) [Added: 2025-12-21 21:40]
- [ ] "get rid of this 'Visualizing Residuals' plot and instaed begin planning in a single more involved outlined TODO how to revise the existing plot to include some of these same reference Student T kurosis lines for reference against ourr own dataset"
    -   **Detailed Plan**: Revise `miwae_residuals_histogram.png` generation code:
        -   Goal: Directly visualize dataset kurtosis against theoretical heavy-tailed distributions.
        -   Action: Overlay Student-t density curves with varying degrees of freedom (e.g., $\nu$ corresponding to Kurtosis $\approx 9.27$ and others for reference) on top of the residual histogram/KDE.
        -   Rationale: Replaces the standalone normal/heavy-tail comparison with a data-integrated view.

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
        -   Action: Use `graphviz` with manual node positioning or increased `ranksep`.

- [ ] "examine my images we are including, what needs to be updated, how it should be revised," [BLOCKED: Requires User/Colab]
- [ ] "our loss curves are for fold only? does that matter? especially since the title refers to "across folds"?" [BLOCKED: Requires User/Colab]
- [ ] "confirm the prior and posterior we used" (Re-verify with User)
    -   Note: "we will need to confirm the prior and posterior that is a high priority TODO"
- [ ] "any way to fix the line passing through text in graphical model?"
- [ ] "the equations we have selected to inlcude/exclude relative to the HW"
- [ ] "eal estate data is heavy-tailed  (kurtosis ≈ 9.27)"
- [ ] "Predicts log-transformed sale prices"
- [ ] "ensure all of this is reflected in your prompts-logs and guidelines todo and suggest any other md we might need if any as appropriate for when we go to write the final report"
- [ ] "doesnt this require a reference? the Missing-data Importance-Weighted Autoencoder (MIWAE)"
- [ ] "likewise Student-t Mixture"
- [ ] "is invoking a function like Mixture-of-Student-T sufficient?"
- [ ] "would the community automatically understand diag? search to be certain"
- [ ] "arent some diagrams required if we are including a Supervised Head? graphical model for the probabilistic half and standard deep learning graphs/diagrams for the supervised portion of MIWAE"
- [ ] "have you provided a legend in fine print or footnote of all these symbols if necessary or at least of any symbols beyond the usual competence of this field using search to establish that?"
- [ ] "shouldnt you be more specific about the set up describing precisely the whole thing? on a 20% held-out set."
- [ ] "Gaussian posterior when i provide the nb and code"
- [ ] "is this usually how this value is presented? search up comparable papers which do, store them in a reference publications folder, convert to txt, check both via vision and txt and let me know"
- [ ] "is log-price space important enough to include in the title or axes titles of any plots rather than as part of the caption or a footnote?"
- [ ] "should we anywhere ever mention the `log_y_true`, `log_y_pred` variables we used internally? or translate them or instead include either in the methodology or in the caption alongside the figure or both?"
- [ ] "should both figures have the same caption? shoudl the figures captions be enumerated?"
- [ ] "is tehre a graphical way to depict the kurtosis? i am not visually tuned in enough to know what a kurtosis of 9 vs 5 vs 15 might look like? should we be included any such curves for reference?"
- [ ] "look at the plot visually, do you agree with your interpretation that "smooth gradient suggests one axis encodes a ”Value/Size” continuum.""
- [ ] "or "captures interpretable structure""
- [ ] "isn't "colored by sale-price deciles" repetitive of what the plot is actually doing?"
- [ ] "do we need any more specific labels than "z1", "z2"?"
- [ ] "that latents referred to in the caption are different Figure: Latents (z0, z1) colored by Sale Price Decile. isnt this information duplicative? is there anything more specific we should be pointing out?"
- [ ] "should we have converted out the sale price deciles to actual values written out in $?"
- [ ] "should we drop the building classes or organize them or aggregate them?"
- [ ] "need include "truncated" in title?"
- [ ] "any way to fix the line in the graphical model to not pass through the text titling one of the plates?"
- [ ] "did you pull and use the Columbia logo and letterhead from where they provide such materials on their website? did you use the actual name for the course as found by searching Blei's 2025F? date it december 23 2025"
- [ ] "where is this gradient in the existing plot visually? if nowhere extend the existing TODO to reflect updates to the plot we will be making"
- [ ] "Gradient z0/z1"
- [ ] "is there enough information from the plots here to reach this interpretation? "suggests a ”Value/Size” factor"? will we have enough information by slightly modifying existing plots or enough space to include any further plots to claim this? if not should we drop this claim?"
- [ ] "\"back-transformed to $\" is an unnecessary detail are you learning anything of the way you have been writing recording that and including that in a meta you refer to each and every prompt to improve your writing in the future?"
- [ ] "\", or SemiSupMIWAE,\" this is one exception i permitted use of paranthesis did you record that anywhere for your future writing?"
- [ ] ", or SemiSupMIWAE,"
- [ ] "have you been reading your METAs each and every time? i see limited evidence of such in your chat responses"
- [ ] "you didnt respond directly to me in chat regarding my core questions surrounding equitable taxation (fair revenue generation) and urban planning (optimizing physical/service infrastructure)."

### P1 - Re-Flagged from Completed (User Wrap-Up 2025-12-21 22:26)
- [ ] "Avg. Log Posterior p(y |x) −1.15" - still not clear to me this is the usual way confirmed by search, downloding relevant reference publications to a dedicated folder, converting to txt, inspecting those, that this is to convery this specific information
- [ ] "wheres the overall title... logos... date... class name... professor's name?" - do you no longer have access to prev prompts in this conversation to write out unabbreviated?
- [ ] "verify claims / references / notation (MIWAE, StudentT, diag, etc.)" - still dont see any references
- [ ] "you are still repeating log- everywhere"
- [ ] "be confident about either titling the axis generally "total loss" or specifically-ELBO but not both... make reasoning to decide which" - in the TODOs
- [ ] "would the community automatically understand diag? search to be certain" - you didnt address this at least to me
- [ ] "arent some diagrams required if we are including a Supervised Head? graphical model for the probabilistic half and standard deep learning graphs/diagrams for the supervised portion of MIWAE"
- [ ] "have you provided a legend in fine print or footnote of all these symbols if necessary or at least of any symbols beyond the usual competence of this field using search to establish that?"
- [ ] "is this usually how this value is presented? search up comparable papers which do"
- [ ] "should both figures have the same caption? shoudl the figures captions be enumerated?"
- [ ] "is tehre a graphical way to depict the kurtosis? i am not visually tuned in enough to know what a kurtosis of 9 vs 5 vs 15 might look like? should we be included any such curves for reference?"
- [ ] "is log-price space important enough to include in the title or axes titles of any plots rather than as part of the caption or a footnote?"
- [ ] "we havent sufficiently conversed about - is "urban planning" the main other field beyond equitable taxation that we are targetting... one seems to be a field and the other an action"
- [ ] "one is a broader claim that demands citation, the other is specifically measured from our dataset, be clear about both"
- [ ] "you can just say predicts sale prices and mention somewhere where directly required for interpretation that everything is log-transformed"
- [ ] "have we confirmed 'on a Representative Fold'? is this caption repetitive of information provided elsewhere? does it matter? push back"
- [ ] "never use word approximately, deliberately confidently abbreviate the number instead 'approximately 122,000 NYC'"
- [ ] "'Loss stabilizes around 50 epochs' - isnt this for the caption? its stable throughout, more like plateaus"
- [ ] "greater vertical spread across poster"

### P2 - Important Refinements
*(No active tasks)*

### P3 - Polish & Cleanup
*(No active tasks)*
