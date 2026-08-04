# DVC Data Versioning

This project uses DVC to version data artifacts that are intentionally excluded from Git.

Tracked data layers:

- `data/bronze.dvc` -> `data/bronze`
- `data/raw.dvc` -> `data/raw`
- `data/silver.dvc` -> `data/silver`
- `data/gold.dvc` -> `data/gold`

Git tracks only DVC metadata files. Actual data files stay ignored by `.gitignore`.

## Common Commands

Check whether local data matches DVC metadata:

```bash
dvc status
```

Update DVC metadata after changing data:

```bash
dvc add data/bronze data/raw data/silver data/gold
git add data/*.dvc
```

Restore data from the local DVC cache:

```bash
dvc checkout
```

## Remote Storage

No DVC remote is configured yet. Add one before relying on `dvc push` / `dvc pull` across machines.

Example local remote:

```bash
dvc remote add -d local_remote /path/outside/repo/nba-scout-assistant-dvc
dvc push
```

Example Google Drive-synced folder remote:

```bash
dvc remote add -d gdrive_sync "$HOME/Library/CloudStorage/GoogleDrive-MyDrive/nba-scout-assistant-dvc"
dvc push
```

## Current Scope

This setup versions data snapshots only. DVC pipeline stages will be added after the local dataset build scripts are finalized.

