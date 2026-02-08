"""
train.py

Experiment driver for running:
- centralized baseline training
- federated simulation (using Server and Client above)
- ablation runs and results export

This script exposes a 'run_experiment' function for programmatic use and a CLI for quick runs.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from tqdm import trange

from medalign.server import Server
from medalign.client import Client
from medalign import configs

logger = logging.getLogger("medalign.train")


def run_federated_simulation(config: Dict[str, Any], output_dir: str = "results/fed_sim"):
    """
    Simple single-machine federated simulation.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    server = Server(config)
    rounds = config.get("federated_rounds", 10)
    clients_per_round = config.get("clients_per_round", 5)
    metrics = {}

    for r in trange(rounds, desc="Federated rounds"):
        server.round = r
        selected = server.select_clients(num_clients=clients_per_round, round_idx=r)
        updates = []
        for cid in selected:
            client = Client(client_id=cid, config=config)
            upd = client.local_train(round_idx=r)
            updates.append(upd)
        server.aggregate(updates, round_idx=r)
        # optional evaluation / logging
        metrics[f"round_{r}"] = {"num_updates": len(updates)}
    # save metrics
    Path(output_dir).joinpath("metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Federated simulation complete. Metrics saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run experiments for medalign.")
    parser.add_argument("--mode", choices=["federated", "central"], default="federated")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/experiment")
    args = parser.parse_args()

    # basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    # load config
    cfg = configs.get_config()
    if args.config:
        # allow a JSON config override file
        import json
        cfg_override = json.loads(Path(args.config).read_text())
        cfg.update(cfg_override)

    if args.mode == "federated":
        run_federated_simulation(cfg, output_dir=args.output_dir)
    else:
        logger.warning("Centralized training mode not implemented in this skeleton.")


if __name__ == "__main__":
    main()
