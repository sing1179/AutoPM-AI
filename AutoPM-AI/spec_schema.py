"""
Structured spec schema for AutoPM-AI.
Output format designed for coding agents to consume.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    source: str
    quote: str
    relevance: str


@dataclass
class UIChange:
    screen: str
    change: str
    component: str | None = None


@dataclass
class DataModelChange:
    entity: str
    change: str
    fields: str | None = None


@dataclass
class Workflow:
    name: str
    steps: list[str]
    edge_cases: list[str] = field(default_factory=list)


@dataclass
class DevTask:
    id: int
    task: str
    type: str  # backend, frontend, migration, config, etc.
    deps: list[int] = field(default_factory=list)


@dataclass
class ProductSpec:
    title: str
    problem: str
    user_story: str
    priority: str
    acceptance_criteria: list[str]
    evidence: list[dict]
    ui_changes: list[dict]
    data_model: list[dict]
    workflows: list[dict]
    dev_tasks: list[dict]
    priority_rationale: str | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "problem": self.problem,
            "user_story": self.user_story,
            "priority": self.priority,
            "priority_rationale": self.priority_rationale,
            "acceptance_criteria": self.acceptance_criteria,
            "evidence": self.evidence,
            "ui_changes": self.ui_changes,
            "data_model": self.data_model,
            "workflows": self.workflows,
            "dev_tasks": self.dev_tasks,
        }

    def to_markdown(self) -> str:
        """Markdown format for coding agent context."""
        lines = [
            f"# {self.title}",
            "",
            "## Problem",
            self.problem,
            "",
            "## User Story",
            self.user_story,
            "",
            "## Priority",
            self.priority,
            "",
        ]
        if self.priority_rationale:
            lines.extend(["## Priority Rationale", self.priority_rationale, ""])
        lines.extend([
            "## Acceptance Criteria",
        ])
        for ac in self.acceptance_criteria:
            lines.append(f"- {ac}")
        lines.extend(["", "## Evidence (Traceability)"])
        for e in self.evidence:
            lines.append(f"- **{e.get('source', 'Unknown')}**: \"{e.get('quote', '')}\" — {e.get('relevance', '')}")
        lines.extend(["", "## UI Changes"])
        for u in self.ui_changes:
            comp = f" ({u.get('component', '')})" if u.get("component") else ""
            lines.append(f"- **{u.get('screen', '')}**: {u.get('change', '')}{comp}")
        lines.extend(["", "## Data Model"])
        for d in self.data_model:
            lines.append(f"- **{d.get('entity', '')}**: {d.get('change', '')}")
        lines.extend(["", "## Workflows"])
        for w in self.workflows:
            lines.append(f"### {w.get('name', '')}")
            for s in w.get("steps", []):
                lines.append(f"- {s}")
            for ec in w.get("edge_cases", []):
                lines.append(f"  - Edge case: {ec}")
        lines.extend(["", "## Dev Tasks (for coding agent)"])
        for t in self.dev_tasks:
            deps = f" (deps: {t.get('deps', [])})" if t.get("deps") else ""
            prio = f" [{t.get('priority', '')}]" if t.get("priority") else ""
            lines.append(f"{t.get('id', 0)}. [{t.get('type', '')}] {t.get('task', '')}{prio}{deps}")
        return "\n".join(lines)


def extract_spec_from_response(response: str) -> dict | None:
    """Extract JSON spec from LLM response. Looks for ```json ... ``` block."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try parsing whole response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return None


def spec_dict_to_markdown(spec: dict) -> str:
    """Convert spec dict to implementation-ready markdown."""
    s = ProductSpec(
        title=spec.get("title", "Untitled"),
        problem=spec.get("problem", ""),
        user_story=spec.get("user_story", ""),
        priority=spec.get("priority", "Medium"),
        acceptance_criteria=spec.get("acceptance_criteria", []),
        evidence=spec.get("evidence", []),
        ui_changes=spec.get("ui_changes", []),
        data_model=spec.get("data_model", []),
        workflows=spec.get("workflows", []),
        dev_tasks=spec.get("dev_tasks", []),
        priority_rationale=spec.get("priority_rationale"),
    )
    return s.to_markdown()
