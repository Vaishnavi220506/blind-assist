# Restore the ASE receipt and register the operating-scope audit

2026-09-22, user-authorized metadata repair. The ledger validator and historical
scientific conclusions are unchanged.

## Cause and exact recovery

`experiments/index.jsonl:303` is the archived ASE run
`ase-body-query-pilot-20260909`. Its input reference
`artifacts.local/downloads/ase-body-query-pilot-20260909/download-receipt.json`
was absent. The validator could not reproduce the aggregate input fingerprint;
the ignored receipt also cannot be recovered from its recorded Git revision.

The original receipt survives at
`artifacts.local/cleanup-records/20260909-approved-storage-cleanup/ase-download-receipt.json`.
Its 179496 bytes have SHA256
`da5fe666d8b65e8df3b75fd8af6cb658d1f40c5535764f7d0437f87bfa3c6d48`.
Combined under the original reference path with the original protocol bytes,
they reproduce the recorded aggregate exactly:
`7a4e60bdce0e4ddde276ff6218a6ff3656acc6e4dd5e59d77d68fad00c55f8a3`.

The receipt was copied byte-for-byte to its original logical path, still under
the canonical artifact junction. The retained cleanup copy remains intact.
No historical row, reference, fingerprint, threshold or validation rule was
edited. All425existing ledger rows pass input validation after recovery.
This restores the registration input receipt, not the old ASE dataset or its
entire evidence tree. It does not restore fresh-data status or rerun the pilot.

## Completed current registration

The prior `unknown terminal id` was a separate missing-record problem: a run ID
is not automatically a decision terminal. Added exactly one terminal and used
the supported commands in dependency order:

1. Create `terminal-dtr-operating-scope-stability-20260922` in the terminal
   registry with the existing report/protocol and source commit
   `6fef145c8a6697ef45158257b5e91f7bf29725bb`.
2. `set-terminal-inheritance` records
   `COMPONENT_OR_CHALLENGER / COMPONENT`, preserving A and the failed stability
   result and its consumed-simulation boundary.
3. `register-experiment` adds `ba-operating-scope-20260922` as **archived**,
   explicitly linked to that terminal. Input identity binds the frozen protocol
   text and the original run's sealed protocol.json, which contains source/code
   hashes. Evaluation seal and independent audit are artifact references.

Both supported commands returned0 and refreshed the decision index. The ledger
now has426rows; the preceding425rows remain byte-identical. This task registers
the requested operating-scope audit only; it does not silently backfill other
historical experiments that also reported the earlier blocker.

## Evidence and validation

Repair evidence: `artifacts.local/work/ba-ledger-repair-20260922/`.
Keep the original failed registration/inheritance receipts in the operating-scope
run, plus the repair's restoration receipt, old ledger/terminal snapshots,
successful command receipts, independent audit and final validation logs.
The existing scientific prediction/evaluation seals remain unchanged.

No source capture, training, dataset download, credential change or continuing
process is involved. Recovery is confined to the existing F:-backed artifact
tree and the new run's registry metadata.
