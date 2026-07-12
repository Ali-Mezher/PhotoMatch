# Contributing to PhotoMatch

Complete workflow for every teammate, from first day to final merge.

---

## One-time setup (do this once)

**1. Install Git**
Download from https://git-scm.com and install with default options.

**2. Install Python 3.11**
Download from https://python.org. During install, check **"Add Python to PATH"**.

**3. Clone the repo**
```bash
git clone https://github.com/Ali-Mezher/PhotoMatch.git
cd PhotoMatch
```

**4. Create your virtual environment and install dependencies**
```bash
python -m venv .venv
```
Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```
Mac/Linux:
```bash
source .venv/bin/activate
```
Then:
```bash
pip install -r requirements.txt
```

**5. Configure your Git identity** (use your own name and email)
```bash
git config user.name "Your-GitHub-Username"
git config user.email "your-email@example.com"
```

---

## Every time you start working on a task

**1. Activate your venv** (every session)
```powershell
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate            # Mac/Linux
```

**2. Switch to `dev` and pull the latest changes**
```bash
git checkout dev
git pull origin dev
```

**3. Create a branch for your task**
Name it after the issue number and what it does:
```bash
git checkout -b feature/issue-2-color-correction
```

Branch naming convention:
| Issue | Branch name |
|-------|-------------|
| #2 | `feature/issue-2-color-correction` |
| #3 | `feature/issue-3-intensity` |
| #6 | `feature/issue-6-face-detector` |
| #8 | `feature/issue-8-faiss-index` |

---

## While working

**Save your progress regularly with commits:**
```bash
git add src/preprocessing/color_geometry.py
git commit -m "Add white balance normalization"
```

Keep commits small and focused — one logical change per commit.

**Push your branch to GitHub** (do this often, not just at the end):
```bash
git push origin feature/issue-2-color-correction
```

---

## When your task is done — open a Pull Request

**1. Push your final changes**
```bash
git push origin feature/issue-2-color-correction
```

**2. Open a Pull Request on GitHub**
- Go to https://github.com/Ali-Mezher/PhotoMatch
- GitHub will show a banner: *"Compare & pull request"* — click it
- Set: **base: `dev`** ← **compare: your branch**
- Write a short description of what you did
- In the description, add `Closes #2` to auto-close the issue when merged
- Assign a teammate as reviewer

**3. Wait for review** — the reviewer will either approve or leave comments

---

## Reviewing a teammate's PR

1. Go to https://github.com/Ali-Mezher/PhotoMatch/pulls
2. Open the PR assigned to you
3. Click **"Files changed"** to read the diff
4. Leave inline comments on anything unclear or wrong
5. If everything looks good: click **"Review changes" → "Approve"**
6. The author then clicks **"Merge pull request"** into `dev`

---

## Keeping your branch up to date with `dev`

If `dev` has new commits while you are working (someone else merged their task), update your branch:
```bash
git checkout dev
git pull origin dev
git checkout feature/your-branch
git merge dev
```
Fix any merge conflicts if they appear, then continue working.

---

## End of project — merging `dev` into `master`

Only done once, when everything is finished and tested:
1. Ali (repo owner) opens a PR from `dev` → `master`
2. All teammates review and approve
3. Merge — this is the final product

---

## Quick reference card

```
Daily start:
  git checkout dev && git pull origin dev
  git checkout -b feature/issue-N-description

While working:
  git add <file>
  git commit -m "Short description"
  git push origin feature/issue-N-description

Done with task:
  → Open PR on GitHub: base=dev, compare=your-branch
  → Add "Closes #N" in PR description
  → Request a reviewer

If dev moved ahead:
  git checkout dev && git pull origin dev
  git checkout feature/your-branch && git merge dev
```
