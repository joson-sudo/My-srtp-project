import json
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


def run_tool_loop(
    client: Any,
    tool_schemas: List[Dict[str, Any]],
    tool_functions: Dict[str, Callable[..., str]],
    user_prompt: str,
    model: str,
    max_steps: int = 5
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]

    for step in range(max_steps):
        logger.info("Agent step %s/%s", step + 1, max_steps)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto"
        )

        ai_message = response.choices[0].message
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        messages.append(ai_message.model_dump())

        if not tool_calls:
            if ai_message.content:
                logger.info("Final response received")
            break

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            handler = tool_functions.get(func_name)
            if not handler:
                result = json.dumps({"status": "error", "message": f"Tool not found: {func_name}"})
            else:
                try:
                    result = handler(**args)
                except Exception as exc:  # pragma: no cover - defensive guard
                    result = json.dumps({"status": "error", "message": str(exc)})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return messages
