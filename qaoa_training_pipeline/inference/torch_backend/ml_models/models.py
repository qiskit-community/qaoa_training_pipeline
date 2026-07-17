import torch.nn as nn

from qaoa_training_pipeline.inference.torch_backend.ml_models.ddpm_transformer import (
    DDPMTransformer,
)
from qaoa_training_pipeline.inference.torch_backend.ml_models.edge_transformer import (
    EdgeTransformer,
)
from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_convolutional_network import (
    GCNModel,
)
from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_isomorphism_network import (
    GINRegression,
)
from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_neural_network import (
    GNNRegression,
)
from qaoa_training_pipeline.inference.torch_backend.ml_models.graph_transformer import (
    GraphTransformer,
)
from qaoa_training_pipeline.inference.torch_backend.ml_models.mlp import MLP


EMBED_DIM = 64
NUM_LAYERS = 4
NUM_HEADS = 4
POS_ENC_DIM = 8
EDGE_DIM = 1


MODEL_REGISTRY = {
    "mlp": lambda input_dim, output_dim, **_: MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
    ),
    "edge_transformer": lambda input_dim, output_dim, **_: EdgeTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        n_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        use_positional_encoding=True,
        permute_nodes_during_training=True,
    ),
    "diffusion_transformer": lambda input_dim, output_dim, timesteps=100, **_: DDPMTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        n_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        timesteps=timesteps,
    ),
    "graph_neural_network": lambda input_dim, output_dim, node_feature_dim=0, **_: GNNRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        edge_dim=EDGE_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        node_feature_dim=node_feature_dim,
    ),
    "graph_isomorphism_network": lambda input_dim, output_dim, node_feature_dim=0, **_: GINRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        edge_dim=EDGE_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        node_feature_dim=node_feature_dim,
    ),
    "graph_transformer": lambda input_dim, output_dim, node_feature_dim=0, **_: GraphTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=EMBED_DIM * 2,
        pos_enc_dim=POS_ENC_DIM,
        node_feature_dim=node_feature_dim,
        random_sign_flip_pe=True,
    ),
    "gcn": lambda input_dim, output_dim, **_: GCNModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=EMBED_DIM,
    ),
}


def get_model(model_type: str, **kwargs) -> nn.Module:
    model_type = model_type.lower()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model type {model_type!r}. " f"Available models: {sorted(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_type](**kwargs)
