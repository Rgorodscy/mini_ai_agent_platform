from sqlalchemy.orm import Session

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
        status: str
    ) -> Execution:
        execution = Execution(
            tenant_id=tenant_id,
            agent_id=agent_id,
            model=model,
            task=task,
            structured_prompt=structured_prompt,
            steps=steps,
            final_response=final_response,
            status=status
        )
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        return execution

    def get_by_agent(
            self, agent_id: str,
            tenant_id: str,
            page: int = 1,
            size: int = 10) -> tuple[list[Execution], int]:
        query = (
            self.db.query(Execution)
            .filter(
                Execution.agent_id == agent_id,
                Execution.tenant_id == tenant_id)
            .order_by(Execution.created_at.desc())
        )
        total = query.count()
        items = query.offset((page - 1) * size).limit(size).all()
        return items, total
