# 📘 CONTRIBUTING.md

This document explains how our team collaborates using **GitHub**, **feature branches**, **pull requests**, **pre-commit hooks**, and **GitHub Actions CI**.
All contributors must follow this workflow.

Here's a summary of the steps needed to contribute.
For further details, keep on reading.

---

# ✔  Process Summary

0. Clone SUSI and install the dev environment
1. Create a branch
2. Modify the code in the new branch. (Write new tests, if possible!)
3. Run tests
4. Commit the changes to the new branch. `pre-commit` will run automatically and format the code
5. Push your branch
6. Open a Pull Request (PR)
7. Continuous Integration (CI) will run in GitHub's servers
8. We will review the PR. If successful, we will merge it
9. Delete your branch

---

# 📥 0. Install SUSI dev environment

1. Clone the latest version of the code.
```bash
git clone <repo-url>
```
2. Navigate into the project folder and install the Python environment.
```bash
uv sync
```
(You need the [`uv`](https://docs.astral.sh/uv/getting-started/installation/)  Python package and environment manager)

3. Activate the Python environment.
```bash
source .venv/bin/activate
```

4. Install the pre-commit hooks (more info on this below)
```bash
pre-commit install
```
You're ready!

(Tip: Since we use **[ruff](https://docs.astral.sh/ruff/)** to format the code, you might consider installng it in your favourite IDE for a better developer experience.)

---

# ➡️ 1. Always pull
Run

```bash
git pull
```
to ensure you have the latest version of the model.

Tip: get used to do this often. You never know when someone has changed the original code.


# 🔀 2. Create a branch

## Our Git Workflow: Feature/Branch
* The `main` branch is **always stable**.
* Every contribution occurs on a feature branch. This means that if you want to add or modify some code, you must create a new branch.
* No commits go directly to `main`. Instead, Pull Requests are created so that every change to `main` is reviewable.

### Branch naming conventions

```
feature/<short-description>
bugfix/<issue-or-description>
refactor/<scope>
```
- Feature: new code behaviour.
- Bugfix: no new behaviour, just fix something that was broken.
- Refactor: no new behaviour, just move things around.

**Examples:**

```
feature/add-simulation-runner
bugfix/fix-numpy-shape-error
```
### How to create a branch
1. Make sure you are in the `main` branch
```bash
git switch main
```

2. Get the latest version of the code
```bash
git pull
```

3. Create the branch and switch to it
```bash
git switch -c feature/my-feature
```
Now you can make the changes you want.
The changes will stay in your local machine until you `push` them.

Since you have branched off of the `main` branch, your changes won't affect the main SUSI code directly, even if you push.
For that to happen, we need to open a Pull Request (see below).

---

# 💱 3. Modify code

You know how to do this!

---

# 🧪 4. Testing
If you are adding features to the code, you would ideally write some tests for them.

We use **pytest** for automated tests.
Run tests locally:

```bash
uv run pytest
```
Or, if your `venv` is activated:

```bash
pytest
```

All tests must pass before submitting a Pull Request.
(This is enforced automatically by the CI with GitHub Actions, see below).

---

# 🧹 4. Commit and `pre-commit`
## Add and Commit the changes in your branch

You can add changes to the code by running
```bash
git add
```
And commit them with
```bash
git commit -m "<your commit message>"
```

## `pre-commit`: Automatic formating
We enforce code formating, linting, import cleanup, and notebook hygiene via **pre-commit**.

The tools we use are:
* **ruff** - Code formatting and linting
* **nbstripout** – Remove jupyter notebook outputs

This means that before you run `git commit`, those tools below will run (locally) and attempt to format the code.
If they can do the formatting automatically, they will.
If there are any errors in the code, they will point them out.
If they cannot format the code and/or fix the errors, you will need to fix them manually.

Your code **must** pass all checks before being pushed.

Note that you can also run pre-commit manually:
```bash
pre-commit run --all-files
```

This is configured in the `.pre-commit-config.yaml` file.


---

# ➡️ 6. Push the changes

Run 
```bash
git push
```
If your local branch does not exist in the remote repository you will get a message suggesting you to do so.

Run it:
```bash
git push --set-upstream origin <your-branch-name> 
```
And after this your branch will exist online too.

Ultimately we want all good changes to be on the `main` branch, but you cannot modify it directly.
For that, we need a Pull Request.

---

# 🔄 7. Pull Requests (PR)

Every merge into `main` happens through a Pull Request.
You cannot `push` directly to the `main` branch.

### Go to github.com and open a PR
Be nice and write a PR that is:

* Small, focused
* Good title and description
* Remember: ideally, tests are included for new functionality

### PR Requirements for approval

* All pre-commit checks pass
* All CI checks pass (see section below)
* One of the admins gives the OK

---

# 🚀 8. CI (GitHub Actions)

Continuous Integration (CI) runs automatically on Pull Requests and on pushes to `main`.
CI checks:

* ruff
* pytest

If CI fails, the PR **cannot** be merged.

This is configured in the `.github/workflows/ci.yaml` file.

---
# 🕵️‍♀️ 9. PR code review

We will review your code.

At least 1 admin needs to approve it.

---

# 🗑️ 10. Delete your branch

First, switch to the `main` branch:

```bash
git switch main
```

Then, delete your branch

```bash
git branch -d <your-branch>
```


