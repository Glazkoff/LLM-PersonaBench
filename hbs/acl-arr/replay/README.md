# Replaying evidence at the appropriate access level

Public, standard-library Python3.10+ (no model calls):

```sh
uv run --no-project python replay/launcher.py cells
```

This verifies mapped aggregate keys, rounding and their manuscript strings. It
does not recalculate scientific estimates or bootstrap intervals. `cells.json`
lists explicit coverage and outstanding unmapped historical results. The older
`aggregates/audit.py` remains available; it was not rerun during this packaging stage.

Restricted scoring plan, from this package directory on an authorized machine:

```sh
uv run --no-project python replay/launcher.py r2 --root /path/to/hbs-unified-20261103 --out /path/to/new-replay
uv run --no-project python replay/launcher.py controls --root /path/to/hbs-unified-20261103 --out /path/to/new-controls
```

These default to plan-only and require the existing restricted tree; they do not
fetch datasets. `--execute` is accepted only inside an existing finite CPU Slurm
allocation and uses the currently selected Python interpreter. Use the pinned
historical environment, not an arbitrary `uv` resolution, for scientific replay.
R2 envelope:4CPU/32GB/2h; controls:4CPU/24GB/20min. No job is submitted by this
launcher. Both outputs are new mode0700 directories; no overwrite or auto-retry.
Controls writes respondent matrices: **the entire output directory is restricted**.
Do not copy it to this public package. Release only reviewed, de-identified aggregates.

The vendored R2 auditor and uncertainty module are byte-identical to the accepted
27183 sources. The gate, original bundle and restricted configuration paths retain
their original hashes and hierarchy; a relocated tree whose configs contain old
absolute paths is not automatically portable. Preserve the mounted path layout or
validate a separate relocation adapter. Public packaging is not anonymous-data
release, inference regeneration or independent validation.

`historical-requirements.txt` and `historical-environment.lock` record the prior
Python environment. They are evidence, not a guarantee of future package availability.
`DEPENDENCIES.md` separates public tools, numerical runtime and restricted inputs.

Offline relocation check,13 November: an11-file minimal package was copied to a
fresh absolute path containing spaces and invoked from an unrelated working
directory. `cells` passed37 checks; both restricted plan-only modes exited0 using
synthetic input directories, the exact pinned gate source and an explicitly invalid
synthetic score receipt. No receipt validation or scientific execution was claimed.
The plan now exposes working_directory, environment_overrides and staged_files,
matching execution rather than hiding the controls import path behind the command.
The caller must not treat NOT_VALIDATED_BY_PLAN_ONLY as execution admission.

Remaining relocation blockers in signed data: CAL_CONFIG.protocol_path,
baseline_run.dir, readout_frozen/readout_receipts[*].path and TEST_CONFIG forecast
spec.path. The gate checks path identity, file hashes and equality with the prior
receipt; the auditor additionally needs sibling twin-20261106-v6 and the original
scheduler history. Changing paths in frozen files would invalidate provenance.
No permissive rewrite or hash exception was added. The package is movable;
arbitrary relocation of the restricted evidence tree is not validated.
