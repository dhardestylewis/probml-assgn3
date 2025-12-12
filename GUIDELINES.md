# Project Guidelines (HW3 & Thesis)

> **CRITICAL**: NEVER perform `git rebase` or `git reset` without explicit permission.

> **META**: All new guidelines must be marked `(UNREVIEWED)` until user confirms review.
> **META**: All new guidelines must include insertion timestamp `[Added: YYYY-MM-DD HH:MM]`.

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

### 1. Fact & Assignment Verification
#### 1.1 Assignment Compliance
- **Verify against `hw3_assignment.txt`**: Ensure every section explicitly answers prompt questions.
- **Fact Verification**: Ensure claims about model performance (RMSE, R^2) match `hw3_verified.txt` or notebook outputs.
- **External Verification**: Use web search to confirm external facts (e.g., NYC tax laws, property codes) if added.

#### 1.2 Source Priority
Prioritize sources in this order:
1. Course materials / Textbook (Murphy)
2. Original methodology papers (MIWAE, VAE)
3. Government/Official sources (NYC Dept. of Planning, NYC Dept. of Finance)
4. Academic domain papers (Real Estate Economics)

### 2. Citation Integrity
#### 2.1 Citation Checking
- **Verify inline citations**: Ensure `\cite{key}` commands have matching `bib` entries.
- **Search**: Use author name, title words, or year to confirm correct key usage.

---

## P2 - Important (Regular Maintenance)

### 3. Bibliography Management
#### 3.1 Formatting Standard
```bibtex
@article{key,

  author = {Name},

  year = {2025},

  title = {Title}

}
```
- Blank line between each field.
- Two-space indentation for fields.

#### 3.2 Organization
Use commented section headers:
- Primary Sources (Methodology)
- Domain Sources (Real Estate, Economics)
- Data Sources (NYC Open Data)

### 4. Documentation
#### 4.1 Session Tracking
- **Prompt Log**: Append conversation prompts to `PROMPTS-LOG.md`.
  - **Start with Timestamp**: Begin every new entry with a timestamp (e.g., `[2025-12-11 21:49]`).
- **Changelog**: Record session changes in `CHANGELOG.md` with date and time.
- **Append ONLY**: Do not revise existing logs; add new entries at the end.
- **Review Status**: Mark historical entries as `(REVIEWED)` and new entries as `(UNREVIEWED)`.

#### 4.2 Version Control
- Check status frequently.
- Commit logically grouped changes.

### 5. TODO Management
#### 5.1 Insertion Policy
- **Context-Aware Insertion**: Do not blindly append new items.
- **Prioritize Immediately**: Assess priority (P1, P2, P3) and insert into the corresponding section.
- **Best Prioritized Place**: Insert logically within the priority group.
- **Renumbering**: Update numbering if inserting into a list.

#### 5.2 Task Archival
- **Move to Completed**: Regularly migrate checked `[x]` items from `TODO.md` to `TODO-COMPLETED.md`.
- **Timestamping**: Append a completion timestamp `[YYYY-MM-DD HH:MM]` to the end of each archived item.
- **Clean Active List**: Keep `TODO.md` focused only on active or pending work.


---

## P3 - Housekeeping

### 6. File Organization
#### 6.1 Naming Conventions
| Suffix | Meaning |
|--------|---------|
| `-SUBMITTED` | Finalized/submitted work |
| `_assignment` | Original assignment prompts/materials |
| `.d/` | LaTeX projects or grouped files |

#### 6.2 Directory Structure
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

#### 6.3 README Requirements
- Add `README.md` to any directory with non-obvious organization.

### 7. Safe Deletion
- Delete files only after verifying content is captured elsewhere.
