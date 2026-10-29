"""Deterministically reduce recorded failing scenarios and traces."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from agent_eval.adapters import result_as_dict
from agent_eval.models import AgentResult, AgentTurnResult, Scenario, TraceEvent
from agent_eval.scoring import score_scenario


class MinimizeError(ValueError):
    """Raised when a failure cannot be reduced or reproduced from recorded data."""


def _aggregate(scenario: Scenario, turns: tuple[AgentTurnResult, ...], fallback: AgentResult) -> AgentResult:
    if not turns:
        return replace(fallback, turns=())
    return replace(
        fallback,
        intent=turns[-1].intent,
        asked=tuple(value for turn in turns for value in turn.asked),
        tools=tuple(value for turn in turns for value in turn.tools),
        actions=tuple(value for turn in turns for value in turn.actions),
        tool_calls=tuple(value for turn in turns for value in turn.tool_calls),
        events=tuple(value for turn in turns for value in turn.events),
        answer=next((turn.answer for turn in reversed(turns) if turn.answer is not None), None),
        notes="\n".join(turn.notes for turn in turns if turn.notes),
        turns=turns,
    )


def _score(scenario: Scenario, result: AgentResult):
    return score_scenario(scenario, result, skip_semantic_judges=True)


def minimize_failure(scenario: Scenario, result: AgentResult) -> dict[str, Any]:
    """Greedily remove turns and trace events while preserving a failure category.

    Reduction is deterministic and yields a 1-minimal result: no remaining single
    turn or event can be removed while retaining the selected diagnostic category.
    The already captured agent trace is scored locally; no provider is called.
    """
    if result.scenario != scenario.name:
        raise MinimizeError(f"result scenario '{result.scenario}' does not match '{scenario.name}'")
    original_score = _score(scenario, result)
    if original_score.passed:
        raise MinimizeError(f"scenario '{scenario.name}' passes; only failing results can be minimized")
    categories = {item.category for item in original_score.diagnostics}
    target_category = next((item.category for item in original_score.diagnostics), "scenario")

    def reproduces(candidate_scenario: Scenario, candidate_result: AgentResult) -> bool:
        score = _score(candidate_scenario, candidate_result)
        return not score.passed and any(item.category == target_category for item in score.diagnostics)

    current_scenario, current_result = scenario, result
    removed_turns: list[int] = []
    removed_events: list[dict[str, Any]] = []

    # A scenario turn and its observed result are one indivisible reduction unit.
    original_indices = list(range(1, min(len(scenario.turns), len(result.turns)) + 1))
    kept_indices = original_indices[:]
    changed = True
    while changed and len(kept_indices) > 1:
        changed = False
        for index in tuple(kept_indices):
            proposed_indices = [value for value in kept_indices if value != index]
            scenario_turns = tuple(scenario.turns[value - 1] for value in proposed_indices)
            observed_turns = tuple(result.turns[value - 1] for value in proposed_indices)
            candidate_scenario = replace(current_scenario, turns=scenario_turns,
                                         input=scenario_turns[0].input)
            candidate_result = _aggregate(candidate_scenario, observed_turns, current_result)
            if reproduces(candidate_scenario, candidate_result):
                kept_indices = proposed_indices
                current_scenario, current_result = candidate_scenario, candidate_result
                removed_turns.append(index)
                changed = True
                break

    if not current_result.turns and (current_result.events or current_result.asked or current_result.actions):
        # Older canonical JSONL captures may have aggregate fields but no nested
        # turn records. Promote that trace to one synthetic turn for event reduction.
        current_result = replace(current_result, turns=(AgentTurnResult(
            intent=current_result.intent, asked=current_result.asked,
            tools=current_result.tools, actions=current_result.actions,
            tool_calls=current_result.tool_calls, events=current_result.events,
            answer=current_result.answer, notes=current_result.notes,
            metadata=current_result.metadata,
        ),))

    # Remove individual ordered trace events and their matching question/action
    # summary fields so the aggregate remains internally consistent.
    turn_event_positions: list[tuple[int, int]] = []
    for turn_index, turn in enumerate(current_result.turns):
        turn_event_positions.extend((turn_index, event_index) for event_index in range(len(turn.events)))
    changed = True
    while changed:
        changed = False
        for turn_index, event_index in tuple(turn_event_positions):
            if turn_index >= len(current_result.turns):
                continue
            turn = current_result.turns[turn_index]
            if event_index >= len(turn.events):
                continue
            event = turn.events[event_index]
            remaining_events = turn.events[:event_index] + turn.events[event_index + 1:]
            if event.event_type == "question":
                asked = list(turn.asked)
                if event.name in asked:
                    asked.remove(event.name)
                candidate_turn = replace(turn, events=remaining_events, asked=tuple(asked))
            elif event.event_type == "action":
                actions = list(turn.actions)
                if event.name in actions:
                    actions.remove(event.name)
                candidate_turn = replace(turn, events=remaining_events, actions=tuple(actions))
            else:
                candidate_turn = replace(turn, events=remaining_events)
            observed = list(current_result.turns)
            observed[turn_index] = candidate_turn
            candidate_result = _aggregate(current_scenario, tuple(observed), current_result)
            if reproduces(current_scenario, candidate_result):
                removed_events.append({"turn": turn_index + 1, "event": asdict(event)})
                current_result = candidate_result
                turn_event_positions = [
                    (turn_no, event_no)
                    for turn_no, item in enumerate(current_result.turns)
                    for event_no in range(len(item.events))
                ]
                changed = True
                break

    final_score = _score(current_scenario, current_result)
    if final_score.passed or target_category not in {item.category for item in final_score.diagnostics}:
        raise MinimizeError("failure could not be reproduced after reduction")
    return {
        "scenario": scenario.name,
        "target_failure_category": target_category,
        "original_failure_categories": sorted(categories),
        "reproduction_verified": True,
        "minimality": "1-minimal under single-turn and single-event deletion",
        "kept_turns": kept_indices,
        "removed_turns": sorted(removed_turns),
        "removed_events": removed_events,
        "original_size": {"turns": len(result.turns),
                          "events": sum(len(turn.events) for turn in result.turns) or len(result.events)},
        "minimized_size": {"turns": len(current_result.turns), "events": sum(len(turn.events) for turn in current_result.turns)},
        "minimized_scenario": asdict(current_scenario),
        "result": result_as_dict(current_result),
        "remaining_failure_categories": sorted({item.category for item in final_score.diagnostics}),
        "first_divergence": asdict(final_score.diagnostics[0]) if final_score.diagnostics else None,
    }
