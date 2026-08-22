import json
from abc import ABC
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, TypeVar

from pydantic import BaseModel

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class AgentResponseError(RuntimeError):
    """Raised when an agent fails to return requested structured result."""


LOG_FILE = Path("agent_full_dump.log")
LOG_FILE_LOCK = Lock()


def dump_agent(label: str, data: Any) -> None:
    if hasattr(data, "model_dump"):
        try:
            data = data.model_dump(mode="json")
        except Exception:
            data = str(data)

    if isinstance(data, (dict, list)):
        body = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    else:
        body = str(data)

    entry = (
        "\n"
        + "=" * 100
        + "\n"
        + f"{datetime.now().isoformat()} | {label}\n"
        + "=" * 100
        + "\n"
        + body
        + "\n"
    )

    with (
        LOG_FILE_LOCK,
        LOG_FILE.open(
            "a",
            encoding="utf-8",
        ) as file,
    ):
        file.write(entry)
        file.flush()


class Agent(ABC):
    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        verbosity: str | None = None,
    ):
        self.llm = llm
        self.model = model
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.verbosity = verbosity

    def _get_structured_output(
        self,
        prompt: str,
        output_type: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Request structured output and validate it against a model."""

        request = {
            "model": self.model,
            "instructions": self.instructions,
            "input": prompt,
            "text_format": output_type,
        }

        if self.verbosity is not None:
            request["text"] = {"verbosity": self.verbosity}

        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens

        if self.reasoning_effort:
            request["reasoning"] = {
                "effort": self.reasoning_effort,
            }

        print("\n" + "=" * 100, flush=True)
        print(
            f"[LLM REQUEST] {datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        print(f"Agent: {type(self).__name__}", flush=True)
        print(f"Output type: {output_type.__name__}", flush=True)
        print(f"Model: {self.model}", flush=True)
        print(f"Reasoning effort: {self.reasoning_effort}", flush=True)
        print(f"Max output tokens: {self.max_output_tokens}", flush=True)

        print("\n--- INSTRUCTIONS ---", flush=True)
        print(self.instructions, flush=True)

        print("\n--- INPUT PROMPT ---", flush=True)
        print(prompt, flush=True)

        print("\n--- OUTPUT SCHEMA ---", flush=True)
        print(
            output_type.model_json_schema(),
            flush=True,
        )

        print("=" * 100, flush=True)

        # Dump exactly what this agent is about to send.
        dump_agent(
            f"{type(self).__name__} REQUEST",
            {
                "agent": type(self).__name__,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "max_output_tokens": self.max_output_tokens,
                "instructions": self.instructions,
                "input_prompt": prompt,
                "output_type": output_type.__name__,
                "output_schema": output_type.model_json_schema(),
                "request": request,
            },
        )

        started = perf_counter()

        try:
            if type(self).__name__ == "OfflineTutorAgent":
                print("[TUTOR STREAM] opening", flush=True)

                raw_parts = []
                delta_count = 0
                character_count = 0

                with self.llm.responses.stream(**request) as stream:
                    print("[TUTOR STREAM] opened", flush=True)

                    for event in stream:
                        if event.type != "response.output_text.delta":
                            continue

                        delta = event.delta

                        raw_parts.append(delta)
                        delta_count += 1
                        character_count += len(delta)

                        if delta_count % 100 == 0:
                            print(
                                "[TUTOR STREAM] "
                                f"{delta_count} deltas, "
                                f"{character_count} characters",
                                flush=True,
                            )

                        if character_count >= 10000:
                            raw_output = "".join(raw_parts)

                            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                            debug_path = f"debug_runaway_tutor_{timestamp}.txt"

                            with open(
                                debug_path,
                                "w",
                                encoding="utf-8",
                            ) as debug_file:
                                debug_file.write(raw_output)

                            dump_agent(
                                f"{type(self).__name__} RUNAWAY STREAM",
                                {
                                    "delta_count": delta_count,
                                    "character_count": character_count,
                                    "raw_output": raw_output,
                                    "request": request,
                                },
                            )

                            print(
                                "\n[TUTOR STREAM] RUNAWAY OUTPUT DETECTED",
                                flush=True,
                            )
                            print(
                                f"Deltas: {delta_count}",
                                flush=True,
                            )
                            print(
                                f"Characters: {character_count}",
                                flush=True,
                            )
                            print(
                                f"Saved to: {debug_path}",
                                flush=True,
                            )

                            print(
                                "\n--- FIRST 3000 CHARACTERS ---",
                                flush=True,
                            )
                            print(
                                raw_output[:3000],
                                flush=True,
                            )

                            print(
                                "\n--- LAST 3000 CHARACTERS ---",
                                flush=True,
                            )
                            print(
                                raw_output[-3000:],
                                flush=True,
                            )

                            raise RuntimeError(
                                "Tutor generated more than 10000 characters."
                            )

                    response = stream.get_final_response()

            else:
                response = self.llm.responses.parse(**request)

        except Exception as error:
            elapsed = perf_counter() - started

            dump_agent(
                f"{type(self).__name__} ERROR",
                {
                    "agent": type(self).__name__,
                    "model": self.model,
                    "elapsed_seconds": elapsed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "request": request,
                },
            )

            print("\n" + "!" * 100, flush=True)
            print(
                f"[LLM ERROR] after {elapsed:.2f}s",
                flush=True,
            )
            print(f"Agent: {type(self).__name__}", flush=True)
            print(f"Error type: {type(error).__name__}", flush=True)
            print(f"Error: {error}", flush=True)
            print("!" * 100 + "\n", flush=True)

            raise

        elapsed = perf_counter() - started

        # Dump the complete response object immediately after receipt.
        dump_agent(
            f"{type(self).__name__} RESPONSE",
            {
                "agent": type(self).__name__,
                "model": self.model,
                "elapsed_seconds": elapsed,
                "response": response,
            },
        )

        print("\n" + "-" * 100, flush=True)
        print(
            f"[LLM RESPONSE] "
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"after {elapsed:.2f}s",
            flush=True,
        )
        print(f"Agent: {type(self).__name__}", flush=True)
        print(f"Output type: {output_type.__name__}", flush=True)

        print("\n--- RAW RESPONSE OBJECT ---", flush=True)
        print(
            response.model_dump_json(indent=2),
            flush=True,
        )

        print("-" * 100 + "\n", flush=True)

        parsed = getattr(response, "output_parsed", None)

        if parsed is None:
            raw_output = getattr(response, "output_text", "")

            dump_agent(
                f"{type(self).__name__} PARSE FAILURE",
                {
                    "raw_output": raw_output,
                    "response": response,
                    "request": request,
                },
            )

            raise AgentResponseError(
                "The model returned no parsed structured output. "
                f"Raw output: {raw_output[:500]}"
            )

        return parsed
