from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.usage import Usage
from app.models.execution import Execution


class ExecutionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        tenant_id: str,
        agent_id: str,
        model: str,
        task: str,
        structured_prompt: dict,
        steps: list,
        final_response: str | None,
        status: str,
        usage: Usage | None = None,
    ) -> Execution:
        usage_fields = {}
        if usage is not None:
            usage_fields = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "llm_calls": usage.llm_calls,
                "latency_ms": usage.latency_ms,
                "cost_usd": usage.cost_usd,
                "usage_by_model": usage.as_dict()["by_model"],
            }

        execution = Execution(
            tenant_id=tenant_id,
            agent_id=agent_id,
            model=model,
            task=task,
            structured_prompt=structured_prompt,
            steps=steps,
            final_response=final_response,
            status=status,
            **usage_fields,
        )
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        return execution

    def get_by_agent(
        self,
        agent_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[Execution], int]:
        query = (
            self.db.query(Execution)
            .filter(
                Execution.agent_id == agent_id,
                Execution.tenant_id == tenant_id,
            )
            .order_by(Execution.created_at.desc())
        )
        total = query.count()
        items = query.offset((page - 1) * size).limit(size).all()
        return items, total

    def aggregate_usage(
        self,
        tenant_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[dict, list[dict], list[dict]]:
        """
        Sums usage for one tenant, overall and grouped by model and agent.

        Aggregation runs in the database rather than by loading rows: a busy
        tenant's history is unbounded, and every execution row carries its
        full prompt and step trail.

        Returns (totals, by_model, by_agent).
        """
        filters = [Execution.tenant_id == tenant_id]
        if since is not None:
            filters.append(Execution.created_at >= since)
        if until is not None:
            filters.append(Execution.created_at <= until)

        columns = (
            func.count(Execution.id),
            func.coalesce(func.sum(Execution.prompt_tokens), 0),
            func.coalesce(func.sum(Execution.completion_tokens), 0),
            func.coalesce(func.sum(Execution.total_tokens), 0),
            func.coalesce(func.sum(Execution.llm_calls), 0),
            func.sum(Execution.cost_usd),
            func.avg(Execution.latency_ms),
        )

        row = self.db.query(*columns).filter(*filters).one()
        totals = self._totals(row)

        by_model = [
            {"model": model, **self._totals(rest)}
            for model, *rest in (
                self.db.query(Execution.model, *columns)
                .filter(*filters)
                .group_by(Execution.model)
                .order_by(func.sum(Execution.total_tokens).desc())
                .all()
            )
        ]

        by_agent = [
            {"agent_id": agent_id, **self._totals(rest)}
            for agent_id, *rest in (
                self.db.query(Execution.agent_id, *columns)
                .filter(*filters)
                .group_by(Execution.agent_id)
                .order_by(func.sum(Execution.total_tokens).desc())
                .all()
            )
        ]

        return totals, by_model, by_agent

    @staticmethod
    def _totals(row) -> dict:
        (
            executions,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            llm_calls,
            cost_usd,
            avg_latency_ms,
        ) = row

        return {
            "executions": executions,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "llm_calls": int(llm_calls or 0),
            # Left as None when no row had a price: summing to 0.0 would
            # claim the tenant's usage was free.
            "cost_usd": (
                round(float(cost_usd), 8) if cost_usd is not None else None
            ),
            "avg_latency_ms": (
                int(avg_latency_ms) if avg_latency_ms is not None else None
            ),
        }
