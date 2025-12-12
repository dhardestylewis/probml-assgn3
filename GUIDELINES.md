# Project Guidelines (HW3 & Thesis)

> **CRITICAL**: NEVER perform `git rebase` or `git reset` without explicit permission.
> **CRITICAL**: When appending to `CHANGELOG.md`, always include the current timestamp.

## P1 - Critical (Every Session)

### 0. Version Control Safety
#### 0.1 Strict Prohibitions
- **NO REBASE/RESET**: NEVER perform `git rebase` or `git reset` without explicit, written user permission.
- **Data Loss Prevention**: These commands rewrite history and can cause irreversible data loss.

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
