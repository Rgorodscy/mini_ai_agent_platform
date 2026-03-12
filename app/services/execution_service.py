from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.repositories.agent_repo import AgentRepository
from app.repositories.execution_repo import ExecutionRepository
from app.schemas.execution import (
    RunRequest,
    ExecutionResponse,
    PaginatedExecutionResponse)
from app.core.guardrail import check_prompt_injection
from app.core.prompt_builder import build_prompt
from app.core.execution_loop import run_execution_loop
from app.config import SUPPORTED_MODELS
from app.logger import get_logger

logger = get_logger(__name__)


class ExecutionService:
    def __init__(self, db: Session):
        self.agent_repo = AgentRepository(db)
        self.execution_repo = ExecutionRepository(db)

    def run_agent(
            self,
            agent_id: str,
            tenant_id: str,
            data: RunRequest) -> ExecutionResponse:
        logger.info(f"Run requested | agent={agent_id} tenant={tenant_id} "
                    f"model={data.model}")

        if data.model not in SUPPORTED_MODELS:
            logger.warning(f"Unsupported model requested | model={data.model} "
                           f"tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported model '{data.model}'. "
                    f"Supported: {SUPPORTED_MODELS}")
            )

        agent = self.agent_repo.get_by_id(agent_id, tenant_id)
        if not agent:
            logger.warning(f"Agent not found | agent={agent_id} "
                           f"tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )

        # Guardrail check
        check_prompt_injection(data.task)

        # Build structured prompt
        structured_prompt = build_prompt(agent, data.task)

        result = run_execution_loop(structured_prompt, agent)

        logger.info(f"Execution complete | agent={agent_id} "
                    f"status={result['status']} steps={len(result['steps'])}")

        execution = self.execution_repo.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            model=data.model,
            task=data.task,
            structured_prompt=structured_prompt,
            steps=result["steps"],
            final_response=result["final_response"],
            status=result["status"]
        )

        return ExecutionResponse.model_validate(execution)

    def get_history(
            self,
            agent_id: str,
            tenant_id: str,
            page: int = 1,
            page_size: int = 10) -> PaginatedExecutionResponse:
        agent = self.agent_repo.get_by_id(agent_id, tenant_id)
        if not agent:
            logger.warning(
                f"Agent not found | agent={agent_id} tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )

        items, total = self.execution_repo.get_by_agent(
            agent_id, tenant_id, page, page_size
        )
        logger.info(
            f"History fetched | agent={agent_id} tenant={tenant_id} "
            f"total={total} page={page}")

        return PaginatedExecutionResponse(
            total=total,
            page=page,
            size=page_size,
            executions=[ExecutionResponse.model_validate(e) for e in items]
        )
