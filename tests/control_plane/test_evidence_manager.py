import pytest
from src.control_plane.evidence_manager import EvidenceManager, Evidence, EvidenceType, VerificationStatus

@pytest.mark.asyncio
async def test_fake_pass_rejected(tmp_path):
    em = EvidenceManager(evidence_path=str(tmp_path / "ev"))
    bad = Evidence(type=EvidenceType.TEST_RESULT, data={"passed": True, "tests_run": 0}, source="test")
    with pytest.raises(ValueError):
        await em.store(bad)

@pytest.mark.asyncio
async def test_tests_run_0_rejected(tmp_path):
    em = EvidenceManager(evidence_path=str(tmp_path / "ev2"))
    bad = Evidence(type=EvidenceType.TEST_RESULT, data={"passed": True, "tests_run": 0, "failures": []}, source="test_runner")
    with pytest.raises(ValueError):
        await em.store(bad)

@pytest.mark.asyncio
async def test_valid_evidence_accepted(tmp_path):
    em = EvidenceManager(evidence_path=str(tmp_path / "ev3"))
    good = Evidence(type=EvidenceType.TEST_RESULT, data={"passed": True, "tests_run": 5, "failures": []}, source="test_runner")
    eid = await em.store(good)
    assert eid is not None
    ev = await em.get(eid)
    assert ev.verification_status == VerificationStatus.VERIFIED.value
    assert ev.evidence_schema_version == 2

@pytest.mark.asyncio
async def test_execution_placeholder_rejected(tmp_path):
    import tempfile, pathlib
    em = EvidenceManager(evidence_path=str(tmp_path / "ev4"))
    # P1.16: execution with success true but placeholder code/tests must be rejected — Verifier must not trust passed
    # Create placeholder with proper exit_code and artifact to reach placeholder check
    p = pathlib.Path(tempfile.gettempdir()) / "test_placeholder_artifact.py"
    p.write_text("# Generated code", encoding="utf-8")
    ev = Evidence(type=EvidenceType.EXECUTION_RESULT, data={"success": True, "result": {"success": True, "code": "# Generated code", "tests": []}}, source="agent", artifact_paths=[str(p)], exit_code=0, generated_by_real_execution=True, trust_level=2, experiment_id="exp1", dataset_id="ds1", dataset_hash="hash12345", config_hash="cfg12345", code_commit="abc123def", source_command="run", artifact_hashes={str(p): "dummy"})
    # Need correct hash
    import hashlib
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    ev.artifact_hashes = {str(p): h}
    # New contract: placeholder must raise ValueError (fail-closed) — not trusting passed
    with pytest.raises(ValueError, match="placeholder"):
        await em.store(ev)
    # Also test that verifier does not trust passed=true when artifact missing (independently recomputed)
    ev_bad = Evidence(type=EvidenceType.EXECUTION_RESULT, data={"success": True, "passed": True, "result": {"success": True, "code": "print(1)", "tests": []}}, source="agent")
    with pytest.raises(ValueError):
        await em.store(ev_bad)
    # Valid execution with artifact is allowed (not trusting passed but verifying 7 checks)
    p2 = pathlib.Path(tempfile.gettempdir()) / "test_artifact_valid.py"
    p2.write_text("print('valid')", encoding="utf-8")
    h2 = hashlib.sha256(p2.read_bytes()).hexdigest()[:16]
    ev2 = Evidence(type=EvidenceType.EXECUTION_RESULT, data={"success": True, "metrics": {"ok": True}}, source="agent", artifact_paths=[str(p2)], exit_code=0, generated_by_real_execution=True, trust_level=2, experiment_id="exp1", dataset_id="ds1", dataset_hash="hash12345", config_hash="cfg12345", code_commit="abc123def", source_command="run", artifact_hashes={str(p2): h2})
    eid2 = await em.store(ev2)
    stored2 = await em.get(eid2)
    assert stored2.verification_status in (VerificationStatus.VERIFIED.value, VerificationStatus.UNVERIFIED.value)
