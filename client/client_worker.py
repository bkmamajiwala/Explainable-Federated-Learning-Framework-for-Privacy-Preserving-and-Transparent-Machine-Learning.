"""Client worker subprocess.

Each simulated client runs in its own OS process, loading only its private
partition and connecting to the Flower server over gRPC. This keeps clients
truly isolated from one another and from the server (only model updates and
aggregate metrics are exchanged).

Usage:
    python -m client.client_worker --cid 0
"""

from __future__ import annotations

import argparse
import sys
import time

from common.config import load_config
from common.logging_setup import setup_logging
from privacy.secure_aggregation import build_mask_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="client_worker")
    parser.add_argument("--cid", type=int, required=True, help="Client (factory) id")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config() if args.config is None else load_config(args.config)
    logger = setup_logging(tag=f"CLIENT {args.cid}", log_dir=config.resolve("results/logs"))

    from client.client_app import client_factory
    from flwr.client import start_numpy_client

    from models.ann import build_ann

    # Build the model once to learn the parameter count needed for masks.
    from client.data_loader import load_client_data

    data = load_client_data(config, args.cid)
    model = build_ann(len(data.feature_names), config)
    param_size = int(sum(a.size for a in model.get_parameters()))
    mask_provider = build_mask_provider(config, param_size)

    client = client_factory(args.cid, config, mask_provider)

    # Give the server time to start listening before connecting.
    time.sleep(4.0)
    logger.info("Connecting to server at %s ...", config.section("federated")["server_address"])
    try:
        start_numpy_client(
            server_address=str(config.section("federated")["server_address"]),
            client=client,
            insecure=True,
        )
        logger.info("Client %d finished.", args.cid)
    except Exception as exc:  # noqa: BLE001
        logger.error("Client %d failed: %s", args.cid, exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())