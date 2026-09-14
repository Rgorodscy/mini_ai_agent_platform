from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.database import SessionLocal
from app.repositories.agent_repo import AgentRepository
from app.repositories.execution_repo import ExecutionRepository
from app.schemas.execution import (
    RunRequest,
    ExecutionResponse,
    PaginatedExecutionResponse,
    UsageReportResponse,
)
from datetime import datetime
from typing import Iterator

from app.core.guardrail import check_prompt_injection
from app.core.prompt_builder import build_prompt
from app.core.execution_loop import run_execution_loop, stream_execution_loop
from app.core.llm import ProviderRejectedModel, supported_models
from app.core.stream import format_sse, run_in_thread
from app.core.usage import track_usage
from app.logger import get_logger

logger = get_logger(__name__)


class ExecutionService:
    def __init__(self, db: Session, session_factory=None):
        self.agent_repo = AgentRepository(db)
        self.execution_repo = ExecutionRepository(db)
        # Used by streamed runs, which record themselves from a worker thread
        # that can outlive the request session.
        self.session_factory = session_factory or SessionLocal

    def _prepare_run(
        self, agent_id: str, tenant_id: str, data: RunRequest
    ) -> tuple:
        """
        Validates a run request and builds its prompt.

        Shared by both entry points so the streaming path cannot drift from
        the blocking one on model validation, tenant scoping or the
        guardrail. Everything here raises HTTPException, which matters for
        streaming: these checks run before the response starts, so a bad
        request still gets a real status code instead of a 200 whose body
        opens with an error.
        """
        logger.info(
            f"Run requested | agent={agent_id} tenant={tenant_id} "
            f"model={data.model}"
        )

        available = supported_models()
        if data.model not in available:
            logger.warning(
                f"Unsupported model requested | model={data.model} "
                f"tenant={tenant_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported model '{data.model}'. "
                    f"Supported: {available}"
                ),
            )

        agent = self.agent_repo.get_by_id(agent_id, tenant_id)
        if not agent:
            logger.warning(
                f"Agent not found | agent={agent_id} tenant={tenant_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        check_prompt_injection(data.task)

        return agent, build_prompt(agent, data.task)

    def run_agent(
        self, agent_id: str, tenant_id: str, data: RunRequest
    ) -> ExecutionResponse:
        agent, structured_prompt = self._prepare_run(agent_id, tenant_id, data)

        try:
            # Everything inside this block accrues to one usage total,
            # including the LLM calls the RAG pipeline makes on its own.
            with track_usage() as usage:
                result = run_execution_loop(
                    structured_prompt, agent, tenant_id, model=data.model
                )
        except ProviderRejectedModel as e:
            # Configuration is wrong, not the request: say so plainly rather
            # than letting the global handler return an opaque 500.
            logger.error(f"Provider rejected the model | {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
            )

        logger.info(
            f"Execution complete | agent={agent_id} "
            f"status={result['status']} steps={len(result['steps'])} "
            f"tokens={usage.total_tokens} calls={usage.llm_calls} "
            f"latency_ms={usage.latency_ms} cost_usd={usage.cost_usd}"
        )

        execution = self.execution_repo.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            model=data.model,
            task=data.task,
            structured_prompt=structured_prompt,
            steps=result["steps"],
            final_response=result["final_response"],
            status=result["status"],
            usage=usage,
        )

        return ExecutionResponse.model_validate(execution)

    def stream_agent(
        self, agent_id: str, tenant_id: str, data: RunRequest
    ) -> Iterator[str]:
        """
        Validates the request, then returns a generator of SSE frames.

        Deliberately not a generator function itself. A generator body does
        not run until the first next(), which for a StreamingResponse is
        after the status and headers have already gone out — so validation
        inside one would turn a 400 into a 200 whose body opens with an
        error. Validating here and returning the generator keeps rejections
        as real status codes.
        """
        agent, structured_prompt = self._prepare_run(agent_id, tenant_id, data)

        return self._stream_events(
            agent_id, tenant_id, data, agent, structured_prompt
        )

    def _stream_events(
        self,
        agent_id: str,
        tenant_id: str,
        data: RunRequest,
        agent,
        structured_prompt: dict,
    ) -> Iterator[str]:
        """
        Relays one SSE frame per step, then a `done` frame.

        The producer thread records the execution, not this generator. When a
        client disconnects, the server closes this generator at whichever
        yield it is paused on, so code placed after the loop here never runs:
        an abandoned stream would leave no audit trail and no usage, although
        the model calls had already been made and paid for.

        The final answer is also held back until the row is written.
        Otherwise a client could read the answer, disconnect before `done`,
        and the run's tokens would never reach /usage.
        """
        session_factory = self.session_factory

        def produce(emit):
            # True once a put timed out with nobody reading. A disconnect
            # alone does not set it: the queue simply goes unread, and a run
            # emits too few events to fill it.
            reader_stalled = False

            def relay(item):
                # A client that stops reading must not stop the run from
                # being recorded, so a failed emit is noted, not raised.
                nonlocal reader_stalled
                if reader_stalled:
                    return
                try:
                    emit(item)
                except BrokenPipeError:
                    reader_stalled = True
                    logger.info(
                        f"Client stopped reading; run continues | "
                        f"agent={agent_id}"
                    )

            result = None
            final_step = None

            # One thread, one context: see core/stream.py for why this
            # cannot be a plain generator.
            with track_usage() as usage:
                for event in stream_execution_loop(
                    structured_prompt, agent, tenant_id, model=data.model
                ):
                    if event["type"] != "step":
                        result = event
                    elif event["step"]["type"] == "final_response":
                        final_step = event["step"]
                    else:
                        relay(("step", event["step"]))

            if result is None:
                return

            execution_id = _record_execution(
                session_factory,
                tenant_id=tenant_id,
                agent_id=agent_id,
                model=data.model,
                task=data.task,
                structured_prompt=structured_prompt,
                steps=result["steps"],
                final_response=result["final_response"],
                status=result["status"],
                usage=usage,
            )

            logger.info(
                f"Execution complete (streamed) | agent={agent_id} "
                f"status={result['status']} tokens={usage.total_tokens} "
                f"calls={usage.llm_calls} latency_ms={usage.latency_ms} "
                f"reader_stalled={reader_stalled}"
            )

            if final_step is not None:
                relay(("step", final_step))

            relay(
                (
                    "done",
                    {
                        "execution_id": execution_id,
                        "status": result["status"],
                        "final_response": result["final_response"],
                        **usage.as_dict(),
                    },
                )
            )

        try:
            for name, payload in run_in_thread(produce):
                yield format_sse(name, payload)
        except ProviderRejectedModel as e:
            logger.error(f"Provider rejected the model | {e}")
            yield format_sse("error", {"detail": str(e)})
        except Exception as e:
            logger.error(f"Streamed execution failed | {e}", exc_info=True)
            yield format_sse(
                "error", {"detail": "An unexpected error occurred."}
            )

    def get_usage(
        self,
        tenant_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> UsageReportResponse:
        """Aggregated LLM consumption for one tenant."""
        totals, by_model, by_agent = self.execution_repo.aggregate_usage(
            tenant_id, since, until
        )

        logger.info(
            f"Usage report | tenant={tenant_id} "
            f"executions={totals['executions']} tokens={totals['total_tokens']}"
        )

        return UsageReportResponse(
            tenant_id=tenant_id,
            since=since,
            until=until,
            **totals,
            by_model=by_model,
            by_agent=by_agent,
        )

    def get_history(
        self, agent_id: str, tenant_id: str, page: int = 1, page_size: int = 10
    ) -> PaginatedExecutionResponse:
        agent = self.agent_repo.get_by_id(agent_id, tenant_id)
        if not agent:
            logger.warning(
                f"Agent not found | agent={agent_id} tenant={tenant_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        items, total = self.execution_repo.get_by_agent(
            agent_id, tenant_id, page, page_size
        )
        logger.info(
            f"History fetched | agent={agent_id} tenant={tenant_id} "
            f"total={total} page={page}"
        )

        return PaginatedExecutionResponse(
            total=total,
            page=page,
            size=page_size,
            executions=[ExecutionResponse.model_validate(e) for e in items],
        )


def _record_execution(session_factory, **fields) -> str:
    """
    Writes an execution row with a session of its own and returns its id.

    Called from the streaming producer thread, which can outlive the request
    and therefore its session.
    """
    session = session_factory()
    try:
        return ExecutionRepository(session).create(**fields).id
    finally:
        session.close()
