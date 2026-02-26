"""
Multi-agent system: producers + reviewers.
Producer agents create output; reviewer agents check and improve it.
"""

from typing import Callable

# --- Producer agents ---

ANALYST_SYSTEM = """You are a sharp, no-nonsense PM analyst. Your job is to give straight answers and move the conversation forward.

RULES:
1. **Be direct** — Answer what the PM actually asked. No preamble, no fluff.
2. **Ask clarifying questions when needed** — If the question is vague, ask 1–2 short clarifying questions. Don't guess.
3. **Use the data when you have it** — If documents are uploaded, cite specific evidence. If not, say so and ask them to upload.
4. **Match the ask** — If they want prioritization, give priorities. If they want a summary, summarize.
5. **Keep it concise** — Bullet points over paragraphs. Markdown when helpful.
6. **Never lecture** — Skip the "As a product manager..." stuff. Just help."""

SPEC_WRITER_SYSTEM = """You are an expert product manager creating implementation-ready specs for coding agents (e.g. Claude).

Based on the conversation and uploaded data, produce a structured product spec. Your response MUST include:

1. A brief 2-3 sentence summary (what we're building and why)
2. A JSON code block with this exact structure (use ```json):

```json
{
  "title": "Feature name",
  "problem": "What problem we're solving, 1-2 sentences",
  "user_story": "As a [user], I want [goal] so that [benefit]",
  "priority": "High|Medium|Low",
  "acceptance_criteria": ["Criterion 1", "Criterion 2", ...],
  "evidence": [
    {"source": "filename", "quote": "exact quote from data", "relevance": "why it supports this"}
  ],
  "ui_changes": [
    {"screen": "Screen/Page name", "change": "What to add or change", "component": "optional component"}
  ],
  "data_model": [
    {"entity": "Entity/Table name", "change": "What to add or modify", "fields": "optional field list"}
  ],
  "workflows": [
    {"name": "Workflow name", "steps": ["Step 1", "Step 2"], "edge_cases": ["Edge case 1"]}
  ],
  "dev_tasks": [
    {"id": 1, "task": "Task description", "type": "backend|frontend|migration|config", "deps": []}
  ]
}
```

RULES:
- Cite specific evidence from uploaded files. Include filename and exact quotes.
- UI changes: screens, components, what changes
- Data model: new tables, columns, relationships
- Workflows: step-by-step user flows, edge cases
- Dev tasks: ordered for implementation, use deps for task order
- Be concrete. A coding agent should be able to implement from this."""

# --- Reviewer agents (check producer work) ---

CRITIC_SYSTEM = """You are a senior PM reviewer. Your job is to critique another agent's response.

For the given ORIGINAL RESPONSE, provide a brief critique (3-5 bullet points):

1. **Correctness** — Does it answer what was asked? Any factual errors?
2. **Evidence** — Are claims backed by data? Are citations specific?
3. **Completeness** — Any gaps, missing context, or unanswered parts?
4. **Clarity** — Is it clear and actionable? Any ambiguity?
5. **Improvements** — What should be added, removed, or changed?

Format as markdown. Be specific. If the response is solid, say so briefly. If there are issues, list them clearly."""

SPEC_CRITIC_SYSTEM = """You are a senior PM/eng reviewer. Your job is to critique an implementation spec.

For the given SPEC (markdown + JSON), provide a brief critique (3-5 bullet points):

1. **Evidence traceability** — Does every recommendation cite specific data? Are quotes accurate?
2. **Completeness** — Are UI changes, data model, workflows, and dev tasks all covered? Any missing edge cases?
3. **Consistency** — Do UI changes match the data model? Do workflows align with acceptance criteria?
4. **Implementability** — Can a coding agent execute the dev tasks? Are they ordered correctly? Any missing deps?
5. **Improvements** — What should be added, fixed, or clarified?

Format as markdown. Be specific. If the spec is solid, say so briefly."""

# --- Reviser agents (improve based on critique) ---

REVISER_SYSTEM = """You are a PM who improves work based on feedback.

You have:
- The ORIGINAL RESPONSE from another agent
- A CRITIQUE from a reviewer

Produce an IMPROVED RESPONSE that addresses the critique. Keep what works, fix what doesn't. Be concise. Output only the improved response, no meta-commentary."""

SPEC_REVISER_SYSTEM = """You are a PM who improves specs based on feedback.

You have:
- The ORIGINAL SPEC (markdown + JSON)
- A CRITIQUE from a reviewer

Produce an IMPROVED SPEC that addresses the critique. Keep the same JSON structure. Fix evidence, completeness, consistency, or implementability issues. Output the full improved spec (summary + ```json block)."""


def run_agent(
    client,
    system: str,
    user_content: str,
    temperature: float = 0.3,
) -> str:
    """Run a single agent. Returns its response."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def orchestrate_chat(
    client,
    messages: list[dict],
    data_context: str | None,
) -> tuple[str, str]:
    """
    Multi-agent chat: Analyst → Critic → Reviser.
    Returns (final_response, critique).
    """
    system = ANALYST_SYSTEM
    if data_context:
        system += "\n\n## Available data\n" + data_context
    else:
        system += "\n\n## Available data\nNone. If the user asks for analysis, ask them to upload documents."

    # Build conversation for analyst
    conv_text = "\n\n".join(f"**{m['role']}:** {m['content']}" for m in messages)
    analyst_input = f"## Conversation\n\n{conv_text}\n\nRespond to the latest user message."

    # 1. Analyst produces response
    original = run_agent(client, system, analyst_input, temperature=0.3)

    # 2. Critic reviews
    critic_input = f"## Original response\n\n{original}\n\n## Your critique"
    critique = run_agent(client, CRITIC_SYSTEM, critic_input, temperature=0.2)

    # 3. Reviser improves (only if critic found issues worth fixing)
    reviser_input = f"## Original response\n\n{original}\n\n## Critique\n\n{critique}\n\n## Produce improved response"
    final = run_agent(client, REVISER_SYSTEM, reviser_input, temperature=0.2)

    return final, critique


def orchestrate_spec(
    client,
    messages: list[dict],
    data_context: str | None,
) -> tuple[str, str]:
    """
    Multi-agent spec: Spec Writer → Spec Critic → Spec Reviser.
    Returns (final_spec, critique).
    """
    system = SPEC_WRITER_SYSTEM
    if data_context:
        system += "\n\n## Available data\n" + data_context
    else:
        system += "\n\n## Available data\nNone. Use the conversation context to infer the feature."

    conv_text = "\n\n".join(f"**{m['role']}:** {m['content']}" for m in messages)
    writer_input = f"## Conversation\n\n{conv_text}\n\nProduce the implementation spec."

    # 1. Spec Writer produces spec
    original = run_agent(client, system, writer_input, temperature=0.2)

    # 2. Spec Critic reviews
    critic_input = f"## Original spec\n\n{original}\n\n## Your critique"
    critique = run_agent(client, SPEC_CRITIC_SYSTEM, critic_input, temperature=0.2)

    # 3. Spec Reviser improves
    reviser_input = f"## Original spec\n\n{original}\n\n## Critique\n\n{critique}\n\n## Produce improved spec"
    final = run_agent(client, SPEC_REVISER_SYSTEM, reviser_input, temperature=0.2)

    return final, critique
