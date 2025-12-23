# Thesis Project Guidelines

> **CRITICAL**: NEVER perform `git rebase` or `git reset` without explicit permission.

> **CRITICAL**: NEVER abbreviate user prompts in `PROMPTS-LOG.md`. Record verbatim. If multiple prompts, use `[followed by:]` notation—never `...`.

> **CRITICAL**: When appending to `CHANGELOG.md`, always include the current timestamp (date and time). (UNREVIEWED) [Added: 2025-12-12 00:06]

> **META**: All new guidelines must be marked `(UNREVIEWED)` until user confirms review.
> **META**: All new guidelines must include insertion timestamp `[Added: YYYY-MM-DD HH:MM]`.
> **META**: When syncing external guidelines, MERGE with existing project-specific content—do not overwrite. Review previous commit if needed. (UNREVIEWED) [Added: 2025-12-12 00:06]

> **CRITICAL**: Never use approximate symbols (≈) or hedging language (approximately, roughly, about) in claims. All rounding should be reflected in the number itself (e.g., "9.27" not "≈9.27"). [Added: 2025-12-23 17:10]

---

## Prime Directive for AI Responses (UNREVIEWED) [Added: 2025-12-11 23:48] [Last Updated: 2025-12-08]

> This directive supersedes earlier fragments and applies to all AI assistant interactions with Daniel.

### PD.1 Persona and Difficulty Targeting (UNREVIEWED) [Added: 2025-12-11 23:48]
- **Expert Persona**: Always respond from the persona of a senior/staff-level expert matched to the topic:
  - Staff/principal engineer for systems/CUDA
  - Research scientist or quant lead for ML/finance
  - Senior applied researcher for planning/real estate
- **Assume Senior Level**: Daniel operates at a strong senior level.
- **Difficulty Mix**:
  - ~70% in Daniel's zone of proximal development (ZPD)
  - ~20% one level higher (staff+ / research-lead)
  - ~10% aspirational spikes toward field-leader/PI thinking
- **Calibration**: Target explanations slightly above Daniel's current level while remaining well-supported by training data. Calibrate using Daniel's current message style and past interactions.

### PD.2 Answer Style and Structure (UNREVIEWED) [Added: 2025-12-11 23:48]
- **Technically Precise**: Structured and concise, but not stiff.
- **Match Informal Tone**: Match Daniel's phrasing while keeping technical content at senior/staff level.
- **Concrete Mechanisms**: Prefer equations, implementation details, and mechanisms over vague intuitions.
- **Surface Operational Concerns**: When something matters operationally (performance, numerical stability, governance, institutional incentives/risk), surface it explicitly—not as an aside.
- **Respect System Instructions**: Integrate safety and system constraints into guidance framing.

### PD.3 Calibration Footer (REQUIRED) (UNREVIEWED) [Added: 2025-12-11 23:48]
At the end of each **substantive answer**, include a short meta-calibration block with exactly:
1. **Persona Used**: e.g., "Senior CUDA engineer," "Lead quant researcher," "Senior planning/real-estate ML researcher"
2. **Difficulty Tuning**: ZPD vs stretch vs aspirational proportions
3. **Concrete Suggestion**: One way Daniel's communication or framing could move closer to expert roles (e.g., clearer hypotheses, sharper experiment framing, more explicit metrics/baselines)

### PD.4 Log Directive Snapshots (UNREVIEWED) [Added: 2025-12-11 23:48]
- **Purpose**: Periodic "log directives" act as snapshots of Daniel's communication style and meta-preferences for longitudinal comparison.
- **Trigger**: When Daniel explicitly labels a message as a "log directive" (or indicates it should be treated as a style snapshot), treat it as a snapshot anchor.
- **Temporal Context to Include**:
  - Time of day (morning/afternoon/evening or specific local time)
  - Day of the week
  - Full calendar date (e.g., 2025-12-08)
  - Position within month (early/mid/late)
  - Position within quarter (Q1–Q4 and early/mid/late)
  - Position within year (e.g., "end of year," "start of year")
- **Proactive Prompting**: When Daniel gestures at meta-process questions ("test in a new chat," "log this," "snapshot this," etc.):
  - Remind him he can mark the message as a "log directive" for a longitudinal snapshot
  - Briefly suggest the time-context elements above
- **Mirror Back**: When a log directive is given, mirror back a concise summary of temporal context and purpose for future reference.

### PD.5 Historical Reference and Progress Check-Ins (UNREVIEWED) [Added: 2025-12-11 23:48]
- **Reference Snapshot**: Treat the 2025-12-08 prime-directive conversation as a reference snapshot for future comparisons.
- **Anchor Date**: This prime directive was last substantively updated and discussed on **2025-12-08**.
- **Periodic Check-Ins**: Provide progress check-ins that reference this directive and its last-discussed date, commenting on how Daniel's communication and assistant responses are evolving relative to these goals.

### PD.6 Conversation Improvement Meta-Reflection (UNREVIEWED) [Added: 2025-12-23 16:25]
- **Commit Frequently During Work**: Do not wait for user to ask "have you been saving to git?" Commit after every meaningful change, especially before reorganizations.
- **Proactive GUIDELINES Check**: When user provides analysis with structural implications (masks, contracts, dataflow), proactively check if GUIDELINES already covers these concerns. Update GUIDELINES without prompting.
- **Log Conversation Takeaways Pre-Emptively**: Before wrap-up, proactively log conversation takeaways to PROMPTS-LOG.md. Do not wait for user to ask.
- **Verify Preservation on Reorganizations**: When reorganizing files (especially TODO.md), explicitly verify no content is lost before proceeding. Report item counts (before vs after) and key phrase verification.
- **Front-Load Critical Invariants**: When user provides detailed analysis, extract and surface the single most critical invariant (e.g., "eval-subset-only dataflow") as the organizing principle. State it explicitly in first response.
- **Reference Code When Discussing Code Issues**: When discussing mask or dataflow defects, cite specific file and line ranges to anchor the discussion.

---

## P1 - Critical (Every Session)

### 0. Version Control Safety (UNREVIEWED) [Added: 2025-12-11 23:17]
#### 0.1 Strict Prohibitions (UNREVIEWED) [Added: 2025-12-11 23:17]
- **NO REBASE/RESET**: NEVER perform `git rebase` or `git reset` without explicit, written user permission.
- **Data Loss Prevention**: These commands rewrite history and can cause irreversible data loss.

#### 0.2 Branch Hygiene (UNREVIEWED) [Added: 2025-12-11 23:17]
- **NO SWITCH IF DETACHED**: If in detached HEAD state, do NOT switch branches without explicit user permission. Commit work to a new branch first.
- **Stay on Named Branch**: Avoid working in detached HEAD state. If detached, immediately create a branch from current state.
- **Branch Naming**: Use descriptive names with timestamps: `feature_name_YYYYMMDD` or `session_work_YYYYMMDD`.
- **Verify Before Switch**: Run `git status` before switching branches. Commit or stash all changes first.

#### 0.3 Commit Hygiene (UNREVIEWED) [Added: 2025-12-11 23:17]
- **Verify Staging**: Run `git status` BEFORE every commit to confirm what will be committed.
- **Atomic Commits**: Commit logically grouped changes with descriptive messages.
- **Commit Frequently**: Commit after every meaningful change to avoid losing work.
- **No Dangling Work**: Never leave uncommitted changes at end of session.

#### 0.4 Push Protocol (UNREVIEWED) [Added: 2025-12-11 23:17]
- **Verify Before Push**: Run `git log -1 --name-status` to confirm the commit contains expected files.
- **Confirm Branch**: Run `git branch` to verify you are on the intended branch before pushing.
- **Local == Remote Check**: After push, compare remote branch contents against local to ensure match.
- **Backup First**: Create a backup branch or snapshot before any complex git operations.

#### 0.5 Pull Request Protocol (UNREVIEWED) [Added: 2025-12-11 23:17]
- **Create PR for Major Changes**: Use pull requests for significant work (not quick fixes).
- **Descriptive Title/Body**: PR title should summarize change; body should list what was done.
- **Self-Review First**: Review the PR diff yourself before requesting review.
- **No Direct Merge to Main**: Feature branches should be merged via PR, not direct push.
- **Verify After Merge**: After PR is merged, verify `main` branch contains expected changes.

### 1. Requirement Compliance (UNREVIEWED) [Added: 2025-12-12 00:06]
- **Verify against Guidelines**: Ensure content aligns with `UP Thesis Guidelines` and `UP Outline`.
- **Check Constraints**: Monitor word counts, formatting requirements, and required sections (e.g., Abstract, Introduction types).
- **Project-Specific**: For HW3, verify against `hw3_assignment.txt`. Ensure claims about model performance (RMSE, R²) match `hw3_verified.txt` or notebook outputs.

### 1.5 Evaluation Discipline (UNREVIEWED) [Added: 2025-12-23 16:21]

> **CRITICAL**: Evaluation plots must use **eval-subset-only** data. Never mix `df_pred`-wide arrays with eval-length arrays.

#### 1.5.1 Eval-Subset-Only Dataflow (UNREVIEWED) [Added: 2025-12-23 16:21]
- **Single Construction Point**: Construct `df_eval = df_pred[base_eval_mask].copy()` exactly once.
- **Length Invariant**: All derived arrays (`z`, `mu_log_eval`, `y_log_eval`, `residuals`, etc.) must be eval-length.
- **Enforce via Assertion**: Add `assert len(array) == len(df_eval)` for all derived arrays after `df_eval` construction.
- **Plotting Functions**: Every plotting function accepts `df_eval` plus eval-length arrays; optional plot masks are defined in eval-index space (length `len(df_eval)`), not `df_pred` space.

#### 1.5.2 Mask Discipline (UNREVIEWED) [Added: 2025-12-23 16:21]
- **Single Source of Truth**: Define `base_eval_mask` over `df_pred` rows as the only admissible starting point for evaluation plots.
- **Derived Masks**: All plot-specific masks (e.g., `mask_plot_price`, `mask_plot_class`) must derive from `base_eval_mask`.
- **No Duplicate Mask Blocks**: Define masks once; remove any duplicate or hidden-state-dependent mask definitions.
- **Explicit Labeling**: Any non-eval subset (e.g., "full dataset geometry") must be explicitly labeled and never mixed with eval plots.

#### 1.5.3 Plot Contract Requirements (UNREVIEWED) [Added: 2025-12-23 16:21]
- **Self-Reporting Banner**: Every evaluation figure must include a banner displaying: mask name, `n_total_eval`, `n_used`, and drop reasons.
- **Read from Audit Artifact**: Plot banners should read counts from `eval_audit_df`, not recompute ad hoc.
- **Consistent Binning**: Paired plots (e.g., value hexbin + count hexbin) must use identical binning, axis limits, and gridsize.

#### 1.5.4 Row-Level Audit Artifact (UNREVIEWED) [Added: 2025-12-23 16:21]
- **Single Source of Truth**: Create `eval_audit_df` with one row per `df_eval` row.
- **Required Columns**: `has_z`, `has_price`, `passes_global_filter`, `has_bldg`, `in_plot_price`, `in_plot_class`, `drop_reason`.
- **Persist Alongside Plots**: Save `eval_audit_df` (CSV or Parquet) in the same output directory as plots.

### 2. Fact Verification (UNREVIEWED) [Added: 2025-12-12 00:06]

#### 2.1 External Verification (UNREVIEWED) [Added: 2025-12-12 00:06]
- **Always externally verify factual claims** before marking as correct
- Use web search to confirm dates, names, percentages, and events
- Compare document claims against authoritative sources

#### 2.2 Source Priority (UNREVIEWED) [Added: 2025-12-12 00:06]
Prioritize sources in this order:
1. Government sources (texas.gov, austintexas.gov, capitol.texas.gov, NYC Dept. of Planning, NYC Dept. of Finance)
2. Academic sources (university research, course materials, textbooks like Murphy)
3. Original methodology papers (MIWAE, VAE)
4. Established news organizations (Texas Tribune, KUT)
5. Official organizational sites

#### 2.3 Verification Steps (UNREVIEWED) [Added: 2025-12-12 00:06]
1. Search for claims in target document
2. Check if cited reference exists in references.bib
3. Externally verify factual accuracy via web search
4. Compare authoritative sources vs current citations
5. Update bib entries or inline citations as needed

### 3. Citation Integrity (UNREVIEWED) [Added: 2025-12-11 23:27]

#### 3.1 Citation Checking (UNREVIEWED) [Added: 2025-12-11 23:27]
- **Verify inline citations**: Ensure `\cite{key}` commands have matching `bib` entries.
- **Search**: Use author name, title words, or year to confirm correct key usage.

#### 3.2 Adding New Entries (UNREVIEWED) [Added: 2025-12-12 00:06]
- Use verified external sources
- Note the verification date in comments

---

## P2 - Important (Regular Maintenance)

### 4. Bibliography Management (UNREVIEWED) [Added: 2025-12-11 23:27]

#### 4.1 Formatting Standard (UNREVIEWED) [Added: 2025-12-11 23:27]
```bibtex
@article{key,

  author = {Name},

  year = {2025},

  title = {Title}

}
```
- Blank line after `@type{key,`
- Blank line between each field
- Two-space indentation for fields

#### 4.2 Organization (UNREVIEWED) [Added: 2025-12-11 23:27]
Use commented section headers:
- Primary Sources (Government, Court, Methodology)
- Secondary Sources (Academic, Domain)
- News Sources
- Organizational Sources
- Data Sources (NYC Open Data)

### 5. Documentation (UNREVIEWED) [Added: 2025-12-11 23:27]

#### 5.1 Session Tracking (UNREVIEWED) [Added: 2025-12-11 23:27]
- **Prompt Logging**: Append conversation prompts (or a summary) to `PROMPTS-LOG.md` in the root directory.
- **Timestamping**: Every new insertion in `PROMPTS-LOG.md` MUST be preceded by a header with the current timestamp (e.g., `### N. [YYYY-MM-DD HH:MM] Description`).
- **Full Fidelity**: Never truncate or use ellipses (...). Log the exact full text of the prompt to ensure complete context is preserved.
- **Consistent Structure**: Use a numbered list for each prompt turn.
- **Major Decisions**: Record major decisions in `CHANGELOG.md` with date and time.
- **Append ONLY**: Do not revise existing logs; add new entries at the end.
- **Review Status**: Mark historical entries as `(REVIEWED)` and new entries as `(UNREVIEWED)`.
- Create a `TODO` item for reviewing UNREVIEWED logs.

#### 5.2 Version Control (UNREVIEWED) [Added: 2025-12-11 23:27]
- **Commit**: Commit frequently with descriptive messages.
- **Push**: Push to remote after each logical unit of work.

### 6. TODO Management (UNREVIEWED) [Added: 2025-12-11 23:27]

#### 6.1 Insertion Policy (UNREVIEWED) [Added: 2025-12-11 23:27]
- **Context-Aware Insertion**: Do not blindly append new items (TODOs, guideline sections, or list entries) to the top or bottom of files.
- **Prioritize Immediately**: Assess the priority of the new item (P1, P2, P3, or High Priority) and insert it into the corresponding section or tier.
- **Best Prioritized Place**: Insert the item in the most logical position within its priority group (e.g., grouping similar tasks, respecting dependencies, or ordering by importance).
- **Unified Structure**: Do not create separate "Future" or "Deferred" sections outside the P1/P2/P3 hierarchy. Tag future items (e.g., `[Final Project]`) within their appropriate priority level.
- **Renumbering**: If inserting into a numbered list or structured sequence (like guideline sections), update numbering to maintain consistency.
- **Timestamp Insertion**: Append `[Added: YYYY-MM-DD HH:MM]` to every new item upon insertion.
- [ ] "maintain structure": ensure the item is placed logically within its priority group relative to others.
- [ ] "Verbatim Reflection": When capturing user critiques or requests as TODOs, record the text **verbatim** (quoted exactly). Do NOT summarize, paraphrase, or abbreviate with ellipses (...). **Exception**: You may automatically correct obvious misspellings. (UNREVIEWED) [Added: 2025-12-21 21:45]
- [ ] "Invariant-First Pattern": When writing TODOs for critical sections, add one **Invariant** line per section stating the property that must hold, then add the single assertion that enforces it (e.g., `assert len(array) == len(df_eval)`). (UNREVIEWED) [Added: 2025-12-23 16:22]
- [ ] "Mask Declaration Before Plotting": Before writing any new plot code, state the exact base mask in one line (e.g., `mask = base_eval_mask & mask_has_price`) and require the plot to print that mask name and `n_used` inside the figure. (UNREVIEWED) [Added: 2025-12-23 16:22]


#### 6.2 Completion Policy (UNREVIEWED) [Added: 2025-12-11 23:27]
- **Move to Archive**: When a task is done, move it from `TODO.md` to `TODO-COMPLETED.md`.
- **Maintain Structure**: Place the completed item under its corresponding Priority Header (P1, P2, P3) in `TODO-COMPLETED.md`.
- **Timestamping**: Append the completion timestamp to the item (e.g., `[Completed: YYYY-MM-DD HH:MM]`).
- **Clean Active List**: Keep `TODO.md` focused only on active or pending work.

---

## P3 - Housekeeping

### 7. Style & Rigor (Thesis Standards) (UNREVIEWED) [Added: 2025-12-12 00:06]

#### 7.1 Textual Rules (UNREVIEWED) [Added: 2025-12-12 00:06]
- **Counts and Notation**: Avoid informal `n=X` notation in narrative text. Spell out counts (e.g., "The dataset consists of 122,712 properties").
- **Hyphens & Dashes**: Use plain hyphens or colons/commas. Avoid em-dashes.
- **Abbreviations**: Expand parenthetical lists into plain English.
- **No Parentheticals**: Do not use parenthetical references anywhere. Revise text to integrate information directly into the sentence structure. **Exception**: Parentheses are permitted for introducing abbreviations or acronyms (e.g., "Semi-Supervised MIWAE (SemiSupMIWAE)"). (UNREVIEWED) [Updated: 2025-12-21 22:15]
- **Caption Brevity**: Keep captions strictly descriptive of the artifact. Avoid methodological footnotes (e.g., "back-transformed") or weak interpretations (e.g., "skewed by tails") within the caption itself. (UNREVIEWED) [Added: 2025-12-21 22:15]

#### 7.2 Data Description (UNREVIEWED) [Added: 2025-12-12 00:06]
- **Date Ranges**: Explicitly state the temporal coverage of every data source.
- **Source Specificity**: Distinguish between Open Data downloads and Public Information Requests (PIR).

#### 7.3 Timeline Formatting (UNREVIEWED) [Added: 2025-12-12 00:06]
- **Structure**: Use `I) YYYY Month-Month` structure for timeline sections.
- **Inner Tasks**: Use `MM/DD-MM/DD` format for specific task ranges.

### 8. File Organization (UNREVIEWED) [Added: 2025-12-11 23:27]

#### 8.1 Naming Conventions (UNREVIEWED) [Added: 2025-12-11 23:27]
| Suffix | Meaning |
|--------|---------|
| `-SUBMITTED` | Finalized/submitted work |
| `-TO_INTEGRATE` | Content pending integration |
| `-COMPREHENSIVE` | Complete/long-form documents |
| `-OLD` | Superseded/deprecated content |
| `_assignment` | Original assignment prompts/materials |
| `.d/` | LaTeX projects or grouped files |
| `#-` prefix | Chronological order (e.g., `1-`, `2.1-`, `2.2-`) |
| `*.1, *.2` | Parallel documents (same base number, `.1` = likely first) |

#### 8.2 Directory Structure (UNREVIEWED) [Added: 2025-12-11 23:27]

**Thesis Project:**
```
thesis/
├── Thesis_Draft/
│   └── Thesis_Draft_Reference_Materials/
│       ├── Prompts.d/      # Session prompts by quality tier
│       ├── Background-COMPREHENSIVE.d/
│       └── references.bib
├── Assignments_and_Proposal-SUBMITTED/
├── Deprecated_Writings/    # Numbered chronologically with README
├── TODO.md
├── GUIDELINES.md
└── CHANGELOG.md
```

**HW3/ML Assignment Project:**
```
probml-assgn3/
├── hws/
│   ├── hw2.d/
│   ├── hw3.d/
│   └── hw_deprecated/
├── final_project/
├── TODO.md
├── GUIDELINES.md
├── CHANGELOG.md
└── PROMPTS-LOG.md
```

#### 8.3 README Requirements (UNREVIEWED) [Added: 2025-12-11 23:27]
- Add `README.md` to any directory with non-obvious organization
- Explain numbering schemes, content relationships, or special conventions

### 9. Safe Deletion (UNREVIEWED) [Added: 2025-12-11 23:27]
- Delete files only after verifying content is captured elsewhere
- Track deferred work in `TODO.md` with clear, actionable items

### 10. Session Wrap-Up (UNREVIEWED) [Added: 2025-12-12 00:06]
- **Log Prompts**: Ensure all recent prompts are appended to `PROMPTS-LOG.md` with a timestamp, following full-fidelity rules.
- **Update Changelog**: Add a timestamped entry to `CHANGELOG.md` summarizing key changes, decisions, and completed tasks.
- **Review TODOs**: Verify all completed work is checked off in `TODO.md` and moved to `TODO-COMPLETED.md`.
- **Final Commit & Push**: Stage and commit all project changes with a descriptive message, then push to remote (e.g., `git commit -am "..."; git push`).
