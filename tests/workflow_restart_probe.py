"""Offline child-process fixture: no live LLM or e-commerce requests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.customer_service_agent import CustomerServiceAgent
from api.schemas import ChatRequest, ChatResumeRequest
from policies.after_sale_policy import AfterSalePolicyService
from state.native_checkpoint import WorkflowPersistence
from tools.tool_runtime import ToolRuntime
import test_langgraph_workflow as fixtures


def build_agent(persistence, client=None):
    return CustomerServiceAgent(
        classifier_api_key="", answer_api_key="",
        embedding_client=fixtures.WorkflowEmbeddingClient(),
        after_sale_policy_service=AfterSalePolicyService(ToolRuntime(client or fixtures.WorkflowEcommerceClient())),
        workflow_persistence=persistence,
    )


if __name__ == "__main__":
    mode, root = sys.argv[1], Path(sys.argv[2])
    agent = build_agent(WorkflowPersistence.sqlite(root / "graph.sqlite3", root / "approval.sqlite3"))
    try:
        if mode == "seed":
            pending = agent.chat(ChatRequest(
                session_id="process-restart", runtime_user_id="process-user",
                user_message=f"订单 {fixtures.ORDER_ID} 直接退款",
            ))
            payload = dict(
                session_id=pending.session_id, workflow_id=pending.workflow.workflow_id,
                resume_token=pending.workflow.resume_token, reviewer_id="manager-01",
                reviewer_role="after_sale_manager", decision="approved",
            )
        else:
            result = agent.resume(ChatResumeRequest.model_validate_json(sys.stdin.read()))
            payload = result.model_dump(mode="json")
        print(json.dumps(payload, ensure_ascii=False))
    finally:
        agent.close()
