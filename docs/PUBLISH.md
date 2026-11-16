# Publishing this repository to GitHub

The repository is already initialised and committed locally (`git log` shows the first
commit). Only the remote is missing.

## 1. Create the repository on GitHub

Either use the website — **github.com → New repository** — or the CLI:

```powershell
gh repo create fedbrain-federated-brain-network --public --source . --remote origin --push
```

## 2. Push an existing local repository (no `gh`)

```powershell
git remote add origin https://github.com/Cybermatics-prog/fedbrain-federated-brain-network.git
git branch -M main
git push -u origin main
```

If GitHub asks for a password, use a personal access token as the password
(*Settings → Developer settings → Personal access tokens → Fine-grained tokens*, scope
`Contents: read and write`), or configure SSH:

```powershell
ssh-keygen -t ed25519 -C "your@email"
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub   # add this key to GitHub
git remote set-url origin git@github.com:Cybermatics-prog/fedbrain-federated-brain-network.git
git push -u origin main
```

## 3. What is committed, and what is not

Committed: `fedbrain/`, `web/`, `tests/`, `scripts/`, `configs/`, `docs/` (protocol,
results, diagnostics, research log, figures), the recorded demo trace in
`runs/federation-demo-s42/`, and the English screenshots in `docs/images/`.

Ignored, because they are large, rebuildable or third-party:

| Ignored | Why | Rebuild with |
|---|---|---|
| `data/` (~2 GB) | ABIDE downloads and derived features | `python -m fedbrain.runner download` |
| `artifacts/` | FC caches, exported zips, QA screenshots | `python scripts/build_health_data.py` |
| `runs/*` except the demo trace | per-run checkpoints and predictions | rerun the configs |
| `*.pt`, `tools/` | checkpoints and the Cloudflare binary | — |
| `docs/paper.txt` | third-party article text | — |
| `web/qa/` | browser QA output | `python scripts/check_english_ui.py` |
| `.venv/` | local environment | `.\setup.ps1` |

Check before pushing:

```powershell
git status --porcelain
git ls-files | Measure-Object -Line
"{(git ls-files | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum / 1MB} MB"
```

The tracked tree should stay around 5 MB. If `git status` lists anything under `data/`,
`artifacts/` or `runs/`, `.gitignore` is not being applied — do not commit it.

## 4. Current status

Published at <https://github.com/Cybermatics-prog/fedbrain-federated-brain-network>.
`CITATION.cff` and the README clone command already use that URL, so nothing needs
replacing before the next push:

```powershell
git add -A
git commit -m "docs: describe the change"
.\scripts\publish_github.ps1 -Repo 'Cybermatics-prog/fedbrain-federated-brain-network'
```

`publish_github.ps1` reads `$env:GITHUB_TOKEN` and sends it as a one-shot request header,
so the credential is not written into `.git/config`.

Optional repository settings:

- **About** — description: *Six-site federated brain-network learning on ABIDE-I, with a
  data-flow explorer that shows what leaves each site.*
- **Topics** — `federated-learning`, `brain-connectivity`, `abide`, `pytorch`, `reproducibility`.
- **License** — MIT (already in `LICENSE`).
- Keep the paper-reported 79% out of the description: it is a reference value, not this
  implementation's result.
