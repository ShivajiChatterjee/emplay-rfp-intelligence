"""Explicitly invalidate selected checkpoint fields without touching others."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.field_specs import OFFICIAL_FIELD_LABELS
from workflow.checkpoints import invalidate_checkpoint_fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--field", action="append", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.field) - set(OFFICIAL_FIELD_LABELS))
    if unknown:
        parser.error(f"Unknown official fields: {unknown}")
    payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    fingerprint = payload.get("fingerprint")
    if not fingerprint:
        parser.error("Checkpoint has no fingerprint")
    removed = invalidate_checkpoint_fields(
        args.checkpoint,
        fingerprint,
        set(args.field),
        reason=args.reason,
    )
    print("Invalidated:", ", ".join(sorted(removed)) or "none")
    print("Preserved fields:", len(payload.get("fields", {})) - len(removed))


if __name__ == "__main__":
    main()
