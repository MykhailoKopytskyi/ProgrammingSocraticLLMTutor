from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Event, Thread
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from openai import OpenAI

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class BackendResult(Generic[OutputT]):
    parsed: OutputT
    raw_text: str


class ModelProtocolError(RuntimeError):
    """The model returned content that does not satisfy the required protocol."""


class StructuredModelBackend(ABC):
    @property
    @abstractmethod
    def backend_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def config_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_type: type[OutputT],
        max_output_tokens: int,
    ) -> BackendResult[OutputT]:
        raise NotImplementedError


class OpenAIBackend(StructuredModelBackend):
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature

    @property
    def backend_id(self) -> str:
        return self.model

    @property
    def config_id(self) -> str:
        return (
            f"openai:{self.model}:reasoning={self.reasoning_effort or 'default'}:"
            f"temperature={self.temperature if self.temperature is not None else 'default'}"
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_type: type[OutputT],
        max_output_tokens: int,
    ) -> BackendResult[OutputT]:
        request: dict[str, object] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "text_format": output_type,
            "max_output_tokens": max_output_tokens,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            request["temperature"] = self.temperature

        print("\n" + "=" * 120)
        print(f"[OPENAI REQUEST] model={self.model}")
        print(f"[OUTPUT TYPE] {output_type.__name__}")
        print(f"[MAX OUTPUT TOKENS] {max_output_tokens}")
        print(f"[REASONING] {self.reasoning_effort}")
        print(f"[TEMPERATURE] {self.temperature}")

        print("\n--- SYSTEM PROMPT ---")
        print(system_prompt)

        print("\n--- USER PROMPT ---")
        print(user_prompt)

        print("=" * 120, flush=True)

        response = self.client.responses.parse(**request)
        parsed = getattr(response, "output_parsed", None)
        raw_text = getattr(response, "output_text", "") or ""
        if parsed is None:
            raise ModelProtocolError(
                f"{self.model} returned invalid {output_type.__name__}: {raw_text[:800]}"
            )
        return BackendResult(parsed=parsed, raw_text=raw_text)


class ModalBackend(StructuredModelBackend):
    def __init__(
        self,
        *,
        app_name: str = "socraticrepair-eval-model",
        class_name: str = "SocraticRepairModel",
        backend_id: str = "socraticrepair_epoch2",
    ):
        try:
            import modal
        except ImportError as error:
            raise RuntimeError(
                "Modal is required only for the SocraticRepair evaluation backend. "
                "Install modal before using --tutor socraticrepair."
            ) from error

        self.app_name = app_name
        self.class_name = class_name
        self._backend_id = backend_id

        # Keep exactly one long-lived Modal handle and one long-lived asyncio
        # event loop for all dialogue worker threads. Worker threads submit
        # asynchronous Modal RPCs to this loop; several calls can therefore be
        # in flight at once without each worker creating its own transport/channel.
        # Modal then autos-scales those concurrent inputs across the deployed
        # class container pool (currently max_containers=4).
        self._remote_model = modal.Cls.from_name(app_name, class_name)()
        self._event_loop = asyncio.new_event_loop()
        self._event_loop_started = Event()
        self._event_loop_thread = Thread(
            target=self._run_event_loop,
            name="socraticrepair-modal-rpc",
            daemon=True,
        )
        self._event_loop_thread.start()
        self._event_loop_started.wait()

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def config_id(self) -> str:
        return f"modal:{self.app_name}:{self.class_name}:{self._backend_id}"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_type: type[OutputT],
        max_output_tokens: int,
    ) -> BackendResult[OutputT]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Do not use blocking .remote() directly from ThreadPoolExecutor workers.
        # Modal's async API is multiplexed through one dedicated event loop, so
        # ten dialogue workers can issue concurrent requests without creating
        # ten independent input-plane channels.
        future = asyncio.run_coroutine_threadsafe(
            self._generate_remote(messages, max_output_tokens),
            self._event_loop,
        )
        raw_text = future.result()
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        text = text.removesuffix("```")
        text = text.strip()

        try:
            parsed = output_type.model_validate_json(text)
        except ValidationError as error:
            raise ModelProtocolError(
                f"{self.backend_id} returned invalid {output_type.__name__}: {raw_text[:1000]}"
            ) from error
        return BackendResult(parsed=parsed, raw_text=raw_text)

    async def _generate_remote(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: int,
    ) -> str:
        return await self._remote_model.generate.remote.aio(
            messages,
            max_output_tokens,
        )

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._event_loop)
        self._event_loop_started.set()
        self._event_loop.run_forever()
