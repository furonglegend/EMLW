"""
client.py

Client pipeline skeleton that composes modules:
CAFW -> HSF -> CDE -> PCRL -> RL-Gating -> local update -> compression -> DP clip and upload.

The Client class below provides a minimal local_train() method expected by simulation scripts.
"""

from typing import Dict, Any, Optional, Tuple
import logging
import torch
import torch.nn as nn

# import modules (they should exist in medalign package)
from medalign import configs
from medalign.models import MedAlignModel
from medalign.data import TimeSeriesDataset, make_dataloader
from medalign.cafw import CAFW
from medalign.hsf import CrossAttentionFusion
from medalign.cde import CDE, adjacency_from_similarity
from medalign.pcrl import PrototypeStore, nt_xent_loss
from medalign.compression import quantize_tensor, sparsify_topk, ResidualCompensator
from medalign.privacy import clip_vector

logger = logging.getLogger("medalign.client")


class Client:
    """
    Minimal Client object representing a federated client.
    """
    def __init__(self, client_id: int, config: Optional[Dict[str, Any]] = None):
        self.client_id = client_id
        self.config = config or configs.get_config()
        self.device = torch.device(self.config.get("device", "cpu"))
        # build model (small)
        self.model = MedAlignModel(in_channels=3, encoder_dim=self.config["hidden_dim"],
                                   proj_dim=self.config["proj_dim"], num_classes=self.config["num_classes"]).to(self.device)
        # placeholders for modules
        self.cafw = CAFW(feature_names=["feat0", "feat1", "feat2"], emb_dim=16).to(self.device)
        self.hsf = CrossAttentionFusion(modality_dims=[32, 32], model_dim=64)
        self.cde = CDE(in_dim=64, hidden_dims=[64])
        self.prototype_store = PrototypeStore(num_prototypes=10, dim=self.config["proj_dim"])
        self.residual_comp = ResidualCompensator()
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def local_train(self, round_idx: int = 0) -> Dict[str, Any]:
        """
        Run a local training iteration (or multiple local epochs).
        Returns a dict representing the client update (e.g., weight deltas, prototype deltas, metadata).
        This is intentionally high-level; you should replace the data loading and training loop with real data & losses.
        """
        logger.info("Client %s: starting local_train for round %d", self.client_id, round_idx)
        # TODO: load local data for this client; here we use synthetic data helper for demo
        from medalign.data import synthetic_dataset, make_dataloader
        xs, ys = synthetic_dataset(num_examples=64, seq_len=128, channels=3, n_classes=self.config["num_classes"])
        dl = make_dataloader(xs, ys, batch_size=self.config["batch_size"], augment=True, aug_params=self.config["augmentation"], num_workers=0)

        # training loop (one epoch for demo)
        self.model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(self.device)  # (B, C, T)
            yb = yb.to(self.device)
            logits, proj = self.model(xb, return_proj=True)
            cls_loss = nn.CrossEntropyLoss()(logits, yb)
            # contrastive loss placeholder: pair augmented versions or use proj pairs
            # Here we mock by computing NT-Xent between proj and itself with small noise (demo)
            z_i = proj
            z_j = proj + 0.01 * torch.randn_like(proj)
            cont_loss = nt_xent_loss(z_i, z_j, temperature=0.1)
            loss = cls_loss + 0.1 * cont_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(dl))
        # compute weight delta to upload (simple difference from initial weights)
        # For real system, store previous global weights and compute delta against that.
        delta = {k: (v.detach().cpu() - v.detach().cpu() * 0.0) for k, v in self.model.state_dict().items()}  # placeholder zeros
        # prototype delta
        prev_protos = self.prototype_store.get_prototypes()
        # assign and update prototypes using embeddings from last batch (demo)
        with torch.no_grad():
            assignments = self.prototype_store.assign(proj.detach().cpu())
            self.prototype_store.update_with_batch(proj.detach().cpu(), assignments)
        proto_delta = self.prototype_store.compute_delta(prev_protos)

        # compression: quantize prototype delta as example, produce sparse representation
        proto_q = quantize_tensor(proto_delta, num_bits=self.config["compression"]["quant_bits"])
        proto_sparse = sparsify_topk(proto_delta, fraction=self.config["compression"]["topk_fraction"])

        # DP clipping for the example flattened proto_delta (for demonstration)
        flat_proto = proto_delta.view(-1)
        clipped = clip_vector(flat_proto, self.config["dp"]["clip"]) if self.config["dp"]["enabled"] else flat_proto

        # package the "update" to send to server
        update = {
            "client_id": self.client_id,
            "weight_delta_keys": list(delta.keys()),  # keys present
            "proto_quantized": proto_q,
            "proto_sparse": {"indices": proto_sparse["indices"].cpu(), "values": proto_sparse["values"].cpu()},
            "clipped_proto_norm": float(torch.norm(clipped).item()),
            "local_loss": avg_loss,
            "num_samples": len(xs),
        }
        logger.info("Client %s: finished local_train (loss=%.4f)", self.client_id, avg_loss)
        return update


# Convenience runner used by scripts/run_client_sim.py
def run_local_client(client_id: int = 0, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    client = Client(client_id=client_id, config=config)
    return client.local_train(round_idx=0)
