import pytest

from experimental.src.quantai_architecture_audit import (
    ArchitectureAuditReport,
    AuditFinding,
    QuantAIArchitectureSecurityAudit,
)


def valid_layers() -> dict[str, object]:
    return {
        name: object()
        for name in (
            QuantAIArchitectureSecurityAudit
            .REQUIRED_LAYERS
        )
    }


def valid_dependencies() -> dict[str, list[str]]:
    return {
        "data": ["features"],
        "features": ["intelligence"],
        "intelligence": [
            "models",
            "strategy",
        ],
        "models": ["strategy"],
        "strategy": ["risk"],
        "risk": ["execution"],
        "research": [
            "models",
            "strategy",
        ],
        "monitoring": ["research"],
    }


def valid_security() -> dict[str, bool]:
    return {
        "secrets_externalized": True,
        "api_permissions_restricted": True,
        "audit_logging_enabled": True,
        "state_recovery_supported": True,
        "data_integrity_checks_enabled": True,
        "reproducibility_enabled": True,
    }


def test_finding_validation() -> None:
    finding = AuditFinding(
        "ARCHITECTURE",
        "layer:data",
        True,
        "INFO",
        "OK",
    )

    assert finding.passed is True

    with pytest.raises(ValueError):
        AuditFinding(
            "",
            "name",
            True,
            "INFO",
            "",
        )

    with pytest.raises(TypeError):
        AuditFinding(
            "A",
            "B",
            "true",
            "INFO",
            "",
        )

    with pytest.raises(ValueError):
        AuditFinding(
            "A",
            "B",
            True,
            "BAD",
            "",
        )


def test_report_properties() -> None:
    findings = (
        AuditFinding(
            "A",
            "one",
            True,
            "INFO",
            "",
        ),
        AuditFinding(
            "A",
            "two",
            False,
            "WARNING",
            "",
        ),
        AuditFinding(
            "A",
            "three",
            False,
            "CRITICAL",
            "",
        ),
    )

    report = ArchitectureAuditReport(
        False,
        findings,
    )

    assert report.failed_checks == 2
    assert report.critical_failures == 1
    assert report.to_dict()["failed_checks"] == 2


def test_successful_full_audit() -> None:
    report = (
        QuantAIArchitectureSecurityAudit().audit(
            valid_layers(),
            valid_dependencies(),
            valid_security(),
        )
    )

    assert report.passed is True
    assert report.failed_checks == 0
    assert report.critical_failures == 0


def test_missing_layer_fails() -> None:
    layers = valid_layers()
    layers.pop("risk")

    report = (
        QuantAIArchitectureSecurityAudit().audit(
            layers,
            valid_dependencies(),
            valid_security(),
        )
    )

    assert report.passed is False
    assert report.critical_failures >= 1


def test_forbidden_dependency_fails() -> None:
    dependencies = valid_dependencies()
    dependencies["risk"] = ["strategy"]

    report = (
        QuantAIArchitectureSecurityAudit()
        .audit(
            valid_layers(),
            dependencies,
            valid_security(),
        )
    )

    assert report.passed is False


def test_security_failure_is_critical() -> None:
    security = valid_security()
    security["secrets_externalized"] = False

    report = (
        QuantAIArchitectureSecurityAudit()
        .audit(
            valid_layers(),
            valid_dependencies(),
            security,
        )
    )

    assert report.passed is False
    assert report.critical_failures >= 1


def test_missing_security_control_fails() -> None:
    security = valid_security()
    security.pop("audit_logging_enabled")

    report = (
        QuantAIArchitectureSecurityAudit()
        .audit_security(security)
    )

    assert report.passed is False


def test_layer_audit() -> None:
    report = (
        QuantAIArchitectureSecurityAudit()
        .audit_layers(valid_layers())
    )

    assert report.passed is True

    assert len(report.findings) == len(
        QuantAIArchitectureSecurityAudit
        .REQUIRED_LAYERS
    )


def test_dependency_audit() -> None:
    report = (
        QuantAIArchitectureSecurityAudit()
        .audit_dependencies(
            valid_dependencies()
        )
    )

    assert report.passed is True

    assert len(report.findings) == len(
        QuantAIArchitectureSecurityAudit
        .FORBIDDEN_DEPENDENCY_PAIRS
    )


def test_security_audit() -> None:
    report = (
        QuantAIArchitectureSecurityAudit()
        .audit_security(
            valid_security()
        )
    )

    assert report.passed is True
    assert len(report.findings) == 6


def test_audit_input_validation() -> None:
    auditor = (
        QuantAIArchitectureSecurityAudit()
    )

    with pytest.raises(TypeError):
        auditor.audit(
            "invalid",
            {},
            {},
        )

    with pytest.raises(TypeError):
        auditor.audit(
            {},
            "invalid",
            {},
        )

    with pytest.raises(TypeError):
        auditor.audit(
            {},
            {},
            "invalid",
        )

    with pytest.raises(ValueError):
        auditor.audit(
            {},
            {},
            {},
        )


def test_dependency_validation() -> None:
    auditor = (
        QuantAIArchitectureSecurityAudit()
    )

    with pytest.raises(TypeError):
        auditor.audit_dependencies(
            {"risk": "strategy"}
        )

    with pytest.raises(ValueError):
        auditor.audit_dependencies(
            {"": ["strategy"]}
        )

    with pytest.raises(ValueError):
        auditor.audit_dependencies(
            {"risk": [""]}
        )


def test_security_validation() -> None:
    auditor = (
        QuantAIArchitectureSecurityAudit()
    )

    with pytest.raises(TypeError):
        auditor.audit_security(
            {
                "secrets_externalized": "yes"
            }
        )


def test_audit_serialization() -> None:
    report = (
        QuantAIArchitectureSecurityAudit()
        .audit(
            valid_layers(),
            valid_dependencies(),
            valid_security(),
        )
    )

    payload = report.to_dict()

    assert payload["passed"] is True
    assert "findings" in payload
    assert len(payload["findings"]) > 0


def test_extra_dependency_is_allowed() -> None:
    dependencies = valid_dependencies()
    dependencies["strategy"] = [
        "risk",
        "monitoring",
    ]

    report = (
        QuantAIArchitectureSecurityAudit()
        .audit_dependencies(
            dependencies
        )
    )

    assert report.passed is True