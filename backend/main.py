"""Application entry point for the e-commerce customer service agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.customer_service_agent import CustomerServiceAgent
from api.routes import create_router
from config.settings import workflow_storage_paths
from state.native_checkpoint import WorkflowPersistence


def create_app(
    *,
    agent_factory: Callable[[WorkflowPersistence], CustomerServiceAgent] | None = None,
    persistence_factory: Callable[[], WorkflowPersistence] | None = None,
) -> FastAPI:
    """Assemble the FastAPI application while keeping the entry point thin."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        persistence = (persistence_factory() if persistence_factory is not None
                       else WorkflowPersistence.sqlite(*workflow_storage_paths()))
        try:
            agent = (agent_factory(persistence) if agent_factory is not None
                     else CustomerServiceAgent(workflow_persistence=persistence))
            app.state.agent = agent
            yield
        finally:
            persistence.close()
            app.state.agent = None

    app = FastAPI(title="E-commerce Customer Service Agent", version="0.35.0", lifespan=lifespan)

    # The first version is intended for local development and API verification.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_router(lambda: app.state.agent))
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
