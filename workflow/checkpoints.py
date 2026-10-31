"""Fingerprint-protected field checkpoints for safe production resumption."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from agents.field_specs import OFFICIAL_FIELD_LABELS, get_field_spec
from agents.evidence_verification import (
    claim_evidence_errors,
    reattribute_supported_citations,
)
from agents.field_completeness import enrich_contact_value_from_evidence
from agents.schemas import (
    AddendumChange,
    Evidence,
    ExtractedField,
    FieldRunRecord,
)
from agents.validator import validate_field


_LOCK = threading.Lock()


def workflow_fingerprint(bid_folder: Path, runtime_config: dict[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(OFFICIAL_FIELD_LABELS).encode())
    for path in sorted(p for p in bid_folder.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(bid_folder)).replace("\\", "/").encode())
        digest.update(str(path.stat().st_size).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    project_root = Path(__file__).resolve().parents[1]
    for relative in ("rag/chunking.py", "rag/retrieval.py", "agents/field_specs.py"):
        source = project_root / relative
        if source.exists():
            digest.update(relative.encode())
            digest.update(source.read_bytes())
    config = {
        "cohere_model": "rerank-v5.0-pro",
        "schema_version": 2,
        **(runtime_config or {}),
    }
    digest.update(json.dumps(config, sort_keys=True).encode())
    return digest.hexdigest()


def load_checkpoint(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("Checkpoint fingerprint does not match bid documents/index/config")
    return payload


def save_field_checkpoint(
    path: Path, fingerprint: str, field_name: str, record: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        payload = load_checkpoint(path, fingerprint) if path.exists() else {
            "version": 2, "fingerprint": fingerprint, "fields": {}
        }
        payload["fields"][field_name] = record
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def invalidate_checkpoint_fields(
    path: Path,
    fingerprint: str,
    field_names: set[str],
    *,
    reason: str,
) -> set[str]:
    """Atomically remove only named records while retaining an audit entry."""
    if not field_names:
        return set()
    with _LOCK:
        payload = load_checkpoint(path, fingerprint)
        fields = payload.setdefault("fields", {})
        removed = {name for name in field_names if name in fields}
        for name in removed:
            fields.pop(name)
        payload.setdefault("invalidations", []).append({
            "fields": sorted(removed),
            "reason": reason,
        })
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
        return removed


def reusable_checkpoint_state(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: record
        for name, record in payload.get("fields", {}).items()
        if record.get("operational", {}).get("reusable") is True
        and record.get("operational", {}).get("terminal") is True
    }


def checkpoint_addendum_changes(
    records: dict[str, dict[str, Any]],
) -> list[AddendumChange]:
    """Restore deduplicated validated amendments stored with reusable fields."""

    restored: dict[tuple[str, int | None], AddendumChange] = {}
    for field_name, record in records.items():
        for raw in record.get("addendum_changes", []):
            change = AddendumChange.model_validate(raw)
            if change.field_name != field_name:
                continue
            current = record.get("field", {}).get("value")
            if current != change.updated_value:
                continue
            restored[(change.field_name, change.addendum_number)] = change
    return list(restored.values())


def recover_addendum_changes_from_traces(
    output_dir: Path,
    fingerprint: str,
    records: dict[str, dict[str, Any]],
) -> list[AddendumChange]:
    """Recover legacy checkpoint amendments from same-fingerprint local traces.

    Older checkpoints did not store AddendumChange.  Recovery is deliberately
    strict: the trace fingerprint, current field value, and cited source
    file/page pairs must all agree.  No values are inferred from prose.
    """

    restored = {
        (item.field_name, item.addendum_number): item
        for item in checkpoint_addendum_changes(records)
    }
    for path in sorted(output_dir.glob("workflow_trace_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("checkpoint_fingerprint") != fingerprint:
            continue
        final = payload.get("final_output") or {}
        for raw in final.get("addendum_changes", []):
            try:
                change = AddendumChange.model_validate(raw)
            except Exception:
                continue
            record = records.get(change.field_name)
            if not record or record.get("field", {}).get("value") != change.updated_value:
                continue
            evidence_sources = {
                (item.get("file_name"), item.get("page_number"))
                for item in record.get("evidence", [])
            }
            change_sources = {(item.file, item.page) for item in change.sources}
            if change_sources and not change_sources.issubset(evidence_sources):
                continue
            restored.setdefault((change.field_name, change.addendum_number), change)
    return list(restored.values())


def repair_checkpoint_citations(
    path: Path,
    fingerprint: str,
    field_names: set[str],
) -> set[str]:
    """Repair only provenance for fully supported saved values, without LLM calls."""
    repaired: set[str] = set()
    with _LOCK:
        payload = load_checkpoint(path, fingerprint)
        for field_name in sorted(field_names):
            record = payload.get("fields", {}).get(field_name)
            if not record:
                continue
            field = ExtractedField.model_validate(record["field"])
            evidence = [Evidence.model_validate(item) for item in record.get("evidence", [])]
            if field.value is None or not evidence:
                continue
            field, changed = reattribute_supported_citations(field, evidence)
            if not changed:
                continue
            record["field"] = field.model_dump(mode="json")
            record["claim_evidence"] = claim_evidence_errors(
                field.value, field.citations
            )[0]
            repaired.add(field_name)
        payload.setdefault("citation_repairs", []).append({
            "fields": sorted(repaired),
            "method": "deterministic_claim_level_attribution",
        })
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    return repaired


def normalize_checkpoint_values(
    path: Path,
    fingerprint: str,
    field_names: set[str],
) -> set[str]:
    """Apply lossless schema-shape normalization to saved nonterminal fields."""

    from agents.extractor import normalize_extracted_value

    normalized: set[str] = set()
    with _LOCK:
        payload = load_checkpoint(path, fingerprint)
        for field_name in sorted(field_names):
            record = payload.get("fields", {}).get(field_name)
            if not record or not record.get("field"):
                continue
            field = ExtractedField.model_validate(record["field"])
            revised_value = normalize_extracted_value(field_name, field.value)
            if revised_value == field.value:
                continue
            record["field"] = field.model_copy(
                update={"value": revised_value}
            ).model_dump(mode="json")
            normalized.add(field_name)
        payload.setdefault("checkpoint_normalizations", []).append({
            "fields": sorted(normalized),
            "method": "lossless_official_field_shape_normalization",
        })
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    return normalized


def repair_checkpoint_contact_values(
    path: Path,
    fingerprint: str,
    field_names: set[str],
) -> set[str]:
    """Complete saved contact values only from unambiguous indexed evidence."""

    repaired: set[str] = set()
    if "contact_info" not in field_names:
        return repaired
    with _LOCK:
        payload = load_checkpoint(path, fingerprint)
        record = payload.get("fields", {}).get("contact_info")
        if record and record.get("field"):
            field = ExtractedField.model_validate(record["field"])
            evidence = [Evidence.model_validate(item) for item in record.get("evidence", [])]
            revised, changed = enrich_contact_value_from_evidence(field.value, evidence)
            if changed:
                record["field"] = field.model_copy(
                    update={"value": revised}
                ).model_dump(mode="json")
                repaired.add("contact_info")
        payload.setdefault("checkpoint_contact_repairs", []).append({
            "fields": sorted(repaired),
            "method": "unambiguous_verbatim_submission_address",
        })
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    return repaired


def revalidate_checkpoint_fields(
    path: Path,
    fingerprint: str,
    field_names: set[str],
) -> set[str]:
    """Promote repaired fields only after complete deterministic validation."""
    promoted: set[str] = set()
    with _LOCK:
        payload = load_checkpoint(path, fingerprint)
        for field_name in sorted(field_names):
            record = payload.get("fields", {}).get(field_name)
            if not record or not record.get("operational", {}).get(
                "reconciliation_verified", False
            ):
                continue
            field = ExtractedField.model_validate(record["field"])
            evidence = [Evidence.model_validate(item) for item in record.get("evidence", [])]
            decision = validate_field(
                field=field,
                field_spec=get_field_spec(field_name),
                evidence=evidence,
                retry_count=0,
                max_retries=0,
                semantic_validation=False,
            )
            if decision.status != "passed":
                continue
            record["decision"] = decision.model_dump(mode="json")
            record["operational"] = FieldRunRecord(
                status="passed",
                terminal=True,
                reusable=True,
                reasons=decision.reasons,
                reconciliation_verified=True,
            ).model_dump(mode="json")
            promoted.add(field_name)
        payload.setdefault("checkpoint_revalidations", []).append({
            "fields": sorted(promoted),
            "method": "deterministic_claim_and_citation_validation",
        })
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    return promoted
