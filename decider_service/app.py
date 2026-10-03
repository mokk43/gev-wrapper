from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from decider_service.config import Settings, load_settings


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    release_date: date


class ModelMetadataList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelMetadata]


def create_app(
    settings: Settings,
    *,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=str(settings.backend_url),
            transport=backend_transport,
        ) as backend_client:
            app.state.backend_client = backend_client
            yield

    app = FastAPI(title="TypeSafe-compatible Decider service", lifespan=lifespan)

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def list_models() -> ModelMetadataList:
        release_date = settings.model_release_date
        return ModelMetadataList(
            models=[
                ModelMetadata(
                    name=settings.model_name,
                    description=settings.model_description,
                    release_date=release_date,
                ),
                *(
                    ModelMetadata(
                        name=alias,
                        description=(
                            "Compatibility alias routing to Decider model "
                            f"'{settings.model_name}'."
                        ),
                        release_date=release_date,
                    )
                    for alias in settings.model_aliases
                ),
            ]
        )

    return app


def create_app_from_environment() -> FastAPI:
    return create_app(load_settings())
