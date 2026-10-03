from __future__ import annotations

import argparse

import uvicorn

from decider_service.app import create_app
from decider_service.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TypeSafe-compatible Decider service."
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="validate DECIDER_* settings without starting the server",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.validate_config:
        print("configuration valid")
        return

    uvicorn.run(
        create_app(settings),
        host=str(settings.bind_host),
        port=settings.bind_port,
    )


if __name__ == "__main__":
    main()
