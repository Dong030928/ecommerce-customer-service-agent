"""Application entry point for the e-commerce customer service agent."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.customer_service_agent import CustomerServiceAgent
from api.routes import create_router


agent = CustomerServiceAgent()


def create_app() -> FastAPI:
    """Assemble the FastAPI application while keeping the entry point thin."""

    app = FastAPI(title="E-commerce Customer Service Agent", version="0.22.0")

    # The first version is intended for local development and API verification.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_router(lambda: agent))
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
