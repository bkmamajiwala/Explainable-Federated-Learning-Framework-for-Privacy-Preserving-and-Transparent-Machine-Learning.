"""Command-line entry point.

Usage examples
--------------
    python main.py prepare
    python main.py baselines
    python main.py federated
    python main.py explain
    python main.py cluster
    python main.py experiments
    python main.py all
"""

from __future__ import annotations

import argparse
import sys

from common.config import load_config
from common.logging_setup import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Explainable Federated Learning Framework",
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=[
            "prepare",
            "baselines",
            "federated",
            "explain",
            "cluster",
            "experiments",
            "diagrams",
            "all",
        ],
        help="Which pipeline stage to run.",
    )
    p.add_argument("--config", default=None, help="Path to a custom config.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config() if args.config is None else load_config(args.config)
    logger = setup_logging(tag="MAIN", log_dir=config.resolve("results/logs"))

    try:
        if args.mode in ("prepare", "all"):
            from evaluation.experiments import run_preparation

            run_preparation(config)

        if args.mode in ("baselines", "all"):
            from evaluation.experiments import run_baselines

            run_baselines(config)

        if args.mode in ("federated", "all"):
            from evaluation.experiments import run_federated_experiment

            run_federated_experiment(config)

        if args.mode in ("explain", "all"):
            from evaluation.experiments import run_explanation_stage

            run_explanation_stage(config)

        if args.mode in ("cluster", "all"):
            from evaluation.experiments import run_clustering_stage

            run_clustering_stage(config)

        if args.mode in ("experiments", "all"):
            from evaluation.experiments import run_ablation_studies

            run_ablation_studies(config)

        if args.mode in ("diagrams", "all"):
            from visualization.architecture import generate_all_diagrams

            generate_all_diagrams(config)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level error boundary
        logger.error("Fatal error: %s", exc, exc_info=True)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())