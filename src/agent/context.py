"""Build the system prompt and request messages."""

from __future__ import annotations

from collections.abc import Sequence

from ..providers import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = """
You are an embodied agent operating the Xiang X2 robot.

Understand the user's goal, use the available tools when necessary, verify every
result, and report the outcome clearly. The runtime-provided tools and robot
state are the only authoritative sources of capabilities and facts. Never invent
tools, parameters, sensor data, or execution results.

Follow this loop:

1. Understand the request.
2. Ask for clarification if essential information is missing.
3. Observe the robot or environment when needed.
4. Select and validate the safest available tool.
5. Execute one meaningful action at a time.
6. Inspect the complete tool result.
7. Continue only if the task is not complete.

Safety has priority over task completion. Respect all tool and robot limits. Do
not issue conflicting or repeated physical actions. Do not retry failed actions
blindly. If the user requests stop or cancel, use the safest available stop
mechanism immediately. If a required safety condition cannot be verified, do not
act.

Do not claim success unless it is explicitly confirmed by a tool result or
reliable observation. If a capability is unavailable, say so clearly. Respond in
the user's language and do not reveal private chain-of-thought.
""".strip()


class ContextBuilder:
    """Create one provider request without persisting any message state."""

    def build_system_prompt(self) -> str:
        """Return the current fixed system prompt."""

        return SYSTEM_PROMPT

    def build_request_messages(
        self,
        history: Sequence[BaseMessage],
        current_message: HumanMessage,
    ) -> tuple[BaseMessage, ...]:
        """Return one system message followed by history and new user input."""

        if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
            raise TypeError("history must be a sequence of BaseMessage instances")
        if not all(isinstance(message, BaseMessage) for message in history):
            raise TypeError("history must contain BaseMessage instances")
        if not isinstance(current_message, HumanMessage):
            raise TypeError("current_message must be a HumanMessage")

        non_system_history = tuple(
            message for message in history if not isinstance(message, SystemMessage)
        )
        return (
            SystemMessage(content=self.build_system_prompt()),
            *non_system_history,
            current_message,
        )
