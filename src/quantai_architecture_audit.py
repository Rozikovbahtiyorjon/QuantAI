from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AuditFinding:
    category: str
    name: str
    passed: bool
    severity: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("category must be a non-empty string")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")

        if self.severity not in {"INFO", "WARNING", "CRITICAL"}:
            raise ValueError("invalid severity")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string")


@dataclass(frozen=True)
class ArchitectureAuditReport:
    passed: bool
    findings: tuple[AuditFinding, ...]

    @property
    def critical_failures(self) -> int:
        return sum(
            not finding.passed
            and finding.severity == "CRITICAL"
            for finding in self.findings
        )

    @property
    def failed_checks(self) -> int:
        return sum(
            not finding.passed
            for finding in self.findings
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "critical_failures": self.critical_failures,
            "failed_checks": self.failed_checks,
            "findings": [
                {
                    "category": finding.category,
                    "name": finding.name,
                    "passed": finding.passed,
                    "severity": finding.severity,
                    "message": finding.message,
                }
                for finding in self.findings
            ],
        }


class QuantAIArchitectureSecurityAudit:
    REQUIRED_LAYERS = (
        "data",
        "features",
        "intelligence",
        "models",
        "strategy",
        "risk",
        "execution",
        "research",
        "monitoring",
    )

    FORBIDDEN_DEPENDENCY_PAIRS = (
        ("risk", "strategy"),
        ("execution", "strategy"),
        ("data", "strategy"),
        ("monitoring", "strategy"),
    )

    REQUIRED_SECURITY_CHECKS = (
        "secrets_externalized",
        "api_permissions_restricted",
        "audit_logging_enabled",
        "state_recovery_supported",
        "data_integrity_checks_enabled",
        "reproducibility_enabled",
    )

    def audit(
        self,
        layers: Mapping[str, object],
        dependencies: Mapping[
            str,
            list[str] | tuple[str, ...],
        ],
        security: Mapping[str, bool],
    ) -> ArchitectureAuditReport:
        if not isinstance(layers, Mapping):
            raise TypeError("layers must be a mapping")

        if not isinstance(dependencies, Mapping):
            raise TypeError("dependencies must be a mapping")

        if not isinstance(security, Mapping):
            raise TypeError("security must be a mapping")

        if not layers:
            raise ValueError("layers must not be empty")

        findings: list[AuditFinding] = []

        findings.extend(
            self._audit_layers(layers)
        )

        findings.extend(
            self._audit_dependencies(dependencies)
        )

        findings.extend(
            self._audit_security(security)
        )

        passed = not any(
            not finding.passed
            and finding.severity == "CRITICAL"
            for finding in findings
        )

        return ArchitectureAuditReport(
            passed=passed,
            findings=tuple(findings),
        )

    def audit_layers(
        self,
        layers: Mapping[str, object],
    ) -> ArchitectureAuditReport:
        if not isinstance(layers, Mapping):
            raise TypeError("layers must be a mapping")

        if not layers:
            raise ValueError(
                "layers must not be empty"
            )

        findings = self._audit_layers(layers)

        return self._build_report(findings)

    def audit_dependencies(
        self,
        dependencies: Mapping[
            str,
            list[str] | tuple[str, ...],
        ],
    ) -> ArchitectureAuditReport:
        if not isinstance(dependencies, Mapping):
            raise TypeError(
                "dependencies must be a mapping"
            )

        findings = self._audit_dependencies(
            dependencies
        )

        return self._build_report(findings)

    def audit_security(
        self,
        security: Mapping[str, bool],
    ) -> ArchitectureAuditReport:
        if not isinstance(security, Mapping):
            raise TypeError(
                "security must be a mapping"
            )

        findings = self._audit_security(
            security
        )

        return self._build_report(findings)

    def _audit_layers(
        self,
        layers: Mapping[str, object],
    ) -> list[AuditFinding]:
        findings: list[AuditFinding] = []

        for layer in self.REQUIRED_LAYERS:
            present = (
                layer in layers
                and layers[layer] is not None
            )

            findings.append(
                AuditFinding(
                    category="ARCHITECTURE",
                    name=f"layer:{layer}",
                    passed=present,
                    severity="CRITICAL",
                    message=(
                        "Layer is present."
                        if present
                        else "Required layer is missing."
                    ),
                )
            )

        return findings

    def _audit_dependencies(
        self,
        dependencies: Mapping[
            str,
            list[str] | tuple[str, ...],
        ],
    ) -> list[AuditFinding]:
        findings: list[AuditFinding] = []

        for source, targets in dependencies.items():
            if (
                not isinstance(source, str)
                or not source.strip()
            ):
                raise ValueError(
                    "dependency source must be a "
                    "non-empty string"
                )

            if not isinstance(
                targets,
                (list, tuple),
            ):
                raise TypeError(
                    f"dependencies for '{source}' "
                    "must be a list or tuple"
                )

            for target in targets:
                if (
                    not isinstance(target, str)
                    or not target.strip()
                ):
                    raise ValueError(
                        "dependency target must be "
                        "a non-empty string"
                    )

        for source, target in (
            self.FORBIDDEN_DEPENDENCY_PAIRS
        ):
            actual_targets = dependencies.get(
                source,
                (),
            )

            violation = target in actual_targets

            findings.append(
                AuditFinding(
                    category="DEPENDENCY",
                    name=f"{source}->{target}",
                    passed=not violation,
                    severity="CRITICAL",
                    message=(
                        "Dependency direction is allowed."
                        if not violation
                        else "Forbidden dependency detected."
                    ),
                )
            )

        return findings

    def _audit_security(
        self,
        security: Mapping[str, bool],
    ) -> list[AuditFinding]:
        findings: list[AuditFinding] = []

        for name in self.REQUIRED_SECURITY_CHECKS:
            if name not in security:
                findings.append(
                    AuditFinding(
                        category="SECURITY",
                        name=name,
                        passed=False,
                        severity="CRITICAL",
                        message=(
                            "Required security check "
                            "is missing."
                        ),
                    )
                )
                continue

            value = security[name]

            if not isinstance(value, bool):
                raise TypeError(
                    f"security check '{name}' "
                    "must be a bool"
                )

            findings.append(
                AuditFinding(
                    category="SECURITY",
                    name=name,
                    passed=value,
                    severity="CRITICAL",
                    message=(
                        "Security control is enabled."
                        if value
                        else "Security control is disabled."
                    ),
                )
            )

        return findings

    @staticmethod
    def _build_report(
        findings: list[AuditFinding],
    ) -> ArchitectureAuditReport:
        passed = not any(
            not finding.passed
            and finding.severity == "CRITICAL"
            for finding in findings
        )

        return ArchitectureAuditReport(
            passed=passed,
            findings=tuple(findings),
        )