"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    import asyncio

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    for attempt in range(5):
        try:
            curr_session = session
            uid = f"{user_id}_{attempt}" if attempt > 0 else user_id
            if curr_session is None or attempt > 0:
                curr_session = await runner.session_service.create_session(
                    app_name=app_name, user_id=uid
                )
            final_response = ""
            async for event in runner.run_async(
                user_id=uid, session_id=curr_session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text

            return final_response, curr_session
        except Exception:
            if attempt < 4:
                await asyncio.sleep(5.0 * (attempt + 1))
                continue
            raise
