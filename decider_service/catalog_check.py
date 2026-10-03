from __future__ import annotations

import argparse
import asyncio
import os

import httpx

from decider_service.app import ModelMetadataList


async def fetch_catalog(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 10.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelMetadataList:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        transport=transport,
    ) as client:
        response = await client.get("/v1/models")
        response.raise_for_status()
        return ModelMetadataList.model_validate(response.json())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the public TypeSafe-compatible model catalog."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.getenv("TYPESAFE_API_KEY"))
    args = parser.parse_args()

    catalog = asyncio.run(fetch_catalog(args.base_url, api_key=args.api_key))
    for model in catalog.models:
        print(f"{model.name}\t{model.release_date}\t{model.description}")


if __name__ == "__main__":
    main()
