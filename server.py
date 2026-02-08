"""
server.py

Server skeleton responsible for:
- selecting clients
- receiving client updates (prototypes / weight deltas)
- DP-safe aggregation (using medalign.privacy)
- optional clustering / personalization management and prototype synchronization

This file defines a Server class that simulation scripts can instantiate.
"""

from typing import List, Dict, Any, Optional
import logging
import torch

from medalign import configs
from medalign.privacy import secure_aggregate
from medalign.pcrl import PrototypeStore

logger = logging.getLogger("medalign.server")


class Server:
    """
    Simple federated server simulator. Keeps a global prototype store and a global model placeholder.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or configs.get_config()
        self.device = torch.device(self.config.get("device", "cpu"))
        # global prototype store
        self.prototype_store = PrototypeStore(num_prototypes=10, dim=self.config["proj_dim"])
        # placeholder for global model weights (dict)
        self.global_state = {}
        self.round = 0

    def select_clients(self, num_clients: int = 10, round_idx: int = 0) -> List[int]:
        """
        Select clients for a federated round. For simulation, return integer ids.
        """
        # TODO: implement sampling strategy (random, importance, availability)
        return list(range(num_clients))

    def aggregate(self, client_updates: List[Dict[str, Any]], round_idx: int = 0) -> None:
        """
        Aggregate client updates.
        For demonstration, aggregate prototype deltas using secure_aggregate simulation.
        """
        logger.info("Server: aggregating %d client updates for round %d", len(client_updates), round_idx)
        # collect prototype deltas from quantized forms (here we assume dequantization outside)
        protos = []
        for upd in client_updates:
            try:
                # attempt to reconstruct proto delta if present (quantized dict)
                proto_q = upd.get("proto_quantized")
                if proto_q is not None:
                    # dequantize using medalign.compression.dequantize_tensor if available
                    from medalign.compression import dequantize_tensor
                    proto = dequantize_tensor(proto_q)
                    protos.append(proto.to(self.device))
            except Exception as e:
                logger.warning("Failed to decode proto from client %s: %s", upd.get("client_id"), e)
        if len(protos) == 0:
            logger.info("No protos to aggregate this round.")
            return
        # stack and aggregate with DP noise
        aggregated_proto = secure_aggregate(protos, clip=self.config["dp"]["clip"],
                                            epsilon=self.config["dp"]["epsilon"], delta=self.config["dp"]["delta"])
        # update global prototype store (simple replacement or moving average)
        with torch.no_grad():
            # naive replacement of first n prototypes if shapes match, otherwise use moving average
            gp = self.prototype_store.get_prototypes().to(self.device)
            if aggregated_proto.shape == gp.shape:
                # moving average update
                self.prototype_store.prototypes = 0.9 * gp + 0.1 * aggregated_proto
            else:
                logger.warning("Aggregated proto shape mismatch; skipping prototype update.")

    def save_checkpoint(self, path: str):
        # save minimal server state for resuming
        import os
        os.makedirs(path, exist_ok=True)
        torch.save({"round": self.round, "prototypes": self.prototype_store.get_prototypes()}, f"{path}/server_state.pt")

    def load_checkpoint(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.round = data.get("round", 0)
        protos = data.get("prototypes")
        if protos is not None:
            self.prototype_store.prototypes = protos.to(self.device)
