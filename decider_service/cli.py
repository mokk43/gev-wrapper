from __future__ import annotations

import argparse
from copy import deepcopy

import uvicorn
from uvicorn.config import LOGGING_CONFIG

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

    log_config = deepcopy(LOGGING_CONFIG)
    log_config["loggers"]["decider_service"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    uvicorn.run(
        create_app(settings),
        host=str(settings.bind_host),
        port=settings.bind_port,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
