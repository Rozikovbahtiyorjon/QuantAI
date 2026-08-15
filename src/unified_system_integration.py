from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class IntegrationStageResult:
    name: str
    success: bool
    output: Any = None
    error: Optional[str] = None


@dataclass
class UnifiedSystemResult:
    success: bool
    stages: List[IntegrationStageResult] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def stage_names(self) -> List[str]:
        return [
            stage.name
            for stage in self.stages
        ]

    @property
    def completed_stages(self) -> int:
        return sum(
            stage.success
            for stage in self.stages
        )

    @property
    def failed_stages(self) -> int:
        return sum(
            not stage.success
            for stage in self.stages
        )


class QuantAIUnifiedSystem:
    """
    Deterministic orchestration layer for existing
    QuantAI components.

    The orchestrator does not implement trading logic.
    It executes registered stages in order and passes
    the previous stage output to the next stage.
    """

    def __init__(self) -> None:
        self._stages: List[
            tuple[str, Callable[[Any], Any]]
        ] = []

    def register_stage(
        self,
        name: str,
        handler: Callable[[Any], Any],
    ) -> None:
        if not isinstance(
            name,
            str,
        ) or not name.strip():

            raise ValueError(
                "Stage name must be a non-empty string."
            )

        if not callable(handler):

            raise TypeError(
                "Stage handler must be callable."
            )

        if any(
            stage_name == name
            for stage_name, _ in self._stages
        ):

            raise ValueError(
                f"Stage already registered: {name}"
            )

        self._stages.append(
            (
                name,
                handler,
            )
        )

    @property
    def stage_names(self) -> List[str]:
        return [
            name
            for name, _ in self._stages
        ]

    def run(
        self,
        initial_input: Any = None,
    ) -> UnifiedSystemResult:

        stages: List[
            IntegrationStageResult
        ] = []

        outputs: Dict[str, Any] = {}

        errors: List[str] = []

        current = initial_input

        for name, handler in self._stages:

            try:

                current = handler(
                    current
                )

                outputs[name] = current

                stages.append(
                    IntegrationStageResult(
                        name=name,
                        success=True,
                        output=current,
                    )
                )

            except Exception as exc:

                message = (
                    f"{name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                errors.append(
                    message
                )

                stages.append(
                    IntegrationStageResult(
                        name=name,
                        success=False,
                        error=message,
                    )
                )

                break

        return UnifiedSystemResult(
            success=(
                not errors
                and len(stages)
                == len(self._stages)
            ),
            stages=stages,
            outputs=outputs,
            errors=errors,
        )

    def clear(self) -> None:
        self._stages.clear()


def create_default_integration(
    stages: Mapping[
        str,
        Callable[[Any], Any],
    ],
) -> QuantAIUnifiedSystem:

    if not isinstance(
        stages,
        Mapping,
    ):

        raise TypeError(
            "stages must be a mapping."
        )

    system = QuantAIUnifiedSystem()

    for name, handler in stages.items():

        system.register_stage(
            name,
            handler,
        )

    return system


__all__ = [
    "IntegrationStageResult",
    "UnifiedSystemResult",
    "QuantAIUnifiedSystem",
    "create_default_integration",
]