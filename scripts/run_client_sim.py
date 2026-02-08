#!/usr/bin/env python3
"""
scripts/run_client_sim.py

Simulate multiple clients locally (useful for debugging client pipeline without full server).
This script launches multiple client worker processes and collects short reports.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

# Try to import medalign.client run function
try:
    from medalign.client import run_local_client
except Exception:
    run_local_client = None


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "client_sim.log", mode="a"),
        ],
    )


def client_worker(args_tuple):
    """
    Worker wrapper executed in separate processes.
    Expects a tuple (client_id, config_dict).
    If medalign.client.run_local_client is available it calls it and returns its result.
    Otherwise returns a mocked dictionary.
    """
    client_id, config = args_tuple
    try:
        if run_local_client is not None:
            result = run_local_client(client_id=client_id, config=config)
        else:
            # Mocked behavior: simulate some metrics
            random.seed(client_id + int(config.get("seed", 0)))
            result = {
                "client_id": client_id,
                "status": "ok",
                "local_samples": random.randint(50, 500),
                "train_loss": random.random(),
            }
        return result
    except Exception as e:
        return {"client_id": client_id, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Simulate multiple clients locally.")
    parser.add_argument("--num-clients", type=int, default=8, help="Number of simulated clients.")
    parser.add_argument("--processes", type=int, default=4, help="Number of worker processes.")
    parser.add_argument("--output-dir", type=str, default="results/client_sim", help="Output folder.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    setup_logging(out_dir)
    logging.info("Starting client simulation: num_clients=%d processes=%d", args.num_clients, args.processes)
    random.seed(args.seed)

    # build config to pass to clients
    config = {"seed": args.seed, "data_dir": "data/", "local_epochs": 1}

    pool_inputs = [(i, config) for i in range(args.num_clients)]

    results: List[Dict[str, Any]] = []
    with mp.Pool(processes=args.processes) as pool:
        for res in tqdm(pool.imap_unordered(client_worker, pool_inputs), total=len(pool_inputs)):
            results.append(res)
            logging.info("Client result: %s", res)

    # Save a summary JSON
    (out_dir / "client_sim_results.json").write_text(json.dumps(results, indent=2))
    logging.info("Client simulation finished. Saved results to %s", out_dir / "client_sim_results.json")


if __name__ == "__main__":
    main()
