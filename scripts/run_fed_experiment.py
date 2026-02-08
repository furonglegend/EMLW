#!/usr/bin/env python3
"""
scripts/run_fed_experiment.py

High-level driver for federated experiments.
Assumes existence of medalign.server.Server and medalign.client.Client APIs.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from tqdm import trange

# Try to import medalign package modules - these should be implemented in the repo.
try:
    from medalign import configs
    from medalign.server import Server
    from medalign.client import Client
    from medalign.eval import evaluate_global
except Exception:
    # If medalign is not yet available, fallback to dummy placeholders to keep the script runnable
    configs = None
    Server = None
    Client = None
    evaluate_global = None


def setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_fed_experiment.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    if path.suffix.lower() in (".json",):
        return json.loads(path.read_text())
    else:
        # naive fallback: try to import medalign.configs
        if configs is not None:
            return getattr(configs, "EXPERIMENT_CONFIG", {})
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run federated experiment (high-level driver).")
    parser.add_argument("--config", type=str, required=False, default="configs.py",
                        help="Path to experiment config (json). If not provided falls back to medalign.configs.")
    parser.add_argument("--rounds", type=int, default=100, help="Number of federated rounds.")
    parser.add_argument("--clients-per-round", type=int, default=10, help="Number of clients sampled per round.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="results/fed_run", help="Where to save logs and checkpoints.")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate every N rounds.")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint to resume from.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    setup_logging(out_dir)

    logging.info("Starting federated experiment driver")
    logging.info("Command line args: %s", vars(args))

    # set seeds
    random.seed(args.seed)

    # Load config
    try:
        config = load_config(Path(args.config))
    except Exception as e:
        logging.warning("Failed to load config file, using empty config: %s", e)
        config = {}

    # Merge CLI overrides into config
    config.update({
        "rounds": args.rounds,
        "clients_per_round": args.clients_per_round,
        "seed": args.seed,
        "output_dir": str(out_dir),
    })

    # Initialize server
    if Server is None:
        logging.warning("medalign.server.Server not available. Using dummy server behavior.")
        server = None
    else:
        server = Server(config)

    # Optionally initialize a client factory (used for local simulation)
    if Client is None:
        logging.warning("medalign.client.Client not available. Client behavior will be mocked.")
        client_factory = None
    else:
        client_factory = Client

    # If resuming, ask server to load checkpoint
    if args.resume_checkpoint and server is not None:
        try:
            server.load_checkpoint(args.resume_checkpoint)
            logging.info("Resumed from checkpoint: %s", args.resume_checkpoint)
        except Exception as e:
            logging.error("Failed to load checkpoint: %s", e)

    # Main federated loop
    rounds = config.get("rounds", args.rounds)
    clients_per_round = config.get("clients_per_round", args.clients_per_round)
    eval_every = args.eval_every

    for r in trange(rounds, desc="Federated rounds"):
        logging.info("=== Round %d ===", r + 1)

        # 1) Server selects clients (mocked if server not implemented)
        if server is not None:
            selected_clients = server.select_clients(num_clients=clients_per_round, round_idx=r)
        else:
            # fallback: create list of integer ids
            selected_clients = list(range(clients_per_round))

        client_updates = []
        for client_id in selected_clients:
            # 2) Each client performs local work and returns an update
            try:
                if client_factory is not None:
                    client = client_factory(client_id=client_id, config=config)
                    update = client.local_train(round_idx=r)
                else:
                    # Mocked update: small dict
                    update = {"client_id": client_id, "delta": None, "num_samples": 0}
            except Exception as e:
                logging.exception("Client %s failed: %s", client_id, e)
                continue
            client_updates.append(update)

        # 3) Server aggregates updates
        try:
            if server is not None:
                server.aggregate(client_updates, round_idx=r)
                server.save_checkpoint(out_dir / f"checkpoint_round_{r+1}.pt")
            else:
                # mocked aggregation logging
                logging.info("Aggregated %d client updates (mock)", len(client_updates))
        except Exception as e:
            logging.exception("Aggregation failed at round %d: %s", r + 1, e)

        # 4) Periodic evaluation
        if (r + 1) % eval_every == 0:
            try:
                if server is not None and evaluate_global is not None:
                    metrics = evaluate_global(server.model, config)
                    logging.info("Evaluation metrics at round %d: %s", r + 1, metrics)
                    # persist metrics
                    (out_dir / "metrics.json").write_text(json.dumps({f"round_{r+1}": metrics}, indent=2))
                else:
                    logging.info("Evaluation skipped (server or evaluate_global missing).")
            except Exception as e:
                logging.exception("Evaluation failed at round %d: %s", r + 1, e)

    logging.info("Federated experiment finished. Results in %s", out_dir.absolute())


if __name__ == "__main__":
    main()
