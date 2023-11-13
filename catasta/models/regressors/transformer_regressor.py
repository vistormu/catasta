import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Linear


class TransformerRegressor(Module):
    def __init__(self, *,
                 input_dim: int,
                 output_dim: int,
                 d_model: int,
                 n_heads: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 norm_first: bool = False,
                 batch_first: bool = False,
                 bias: bool = True,
                 ) -> None:
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=n_heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                norm_first=norm_first,
                                                batch_first=batch_first,
                                                bias=bias,
                                                )
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=n_encoder_layers,
                                          )
        decoder_layer = TransformerDecoderLayer(d_model=d_model,
                                                nhead=n_heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                norm_first=norm_first,
                                                batch_first=batch_first,
                                                bias=bias,
                                                )
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer,
                                          num_layers=n_decoder_layers,
                                          )

        self.embedding = Linear(in_features=input_dim,
                                out_features=d_model,
                                bias=bias,
                                )

        self.decoder_embedding = Linear(in_features=input_dim,
                                        out_features=d_model,
                                        bias=bias,
                                        )

        self.linear = Linear(in_features=d_model,
                             out_features=output_dim,
                             bias=bias,
                             )

    def forward(self, source: Tensor, target: Tensor | None = None) -> Tensor:
        if target is None:
            target = source

        source = self.embedding(source)
        target = self.decoder_embedding(target)
        source_positional_encoding = torch.arange(source.shape[1], dtype=torch.float32, device=source.device).unsqueeze(0)
        target_positional_encoding = torch.arange(target.shape[1], dtype=torch.float32, device=target.device).unsqueeze(0)  # type: ignore
        source += source_positional_encoding
        target += target_positional_encoding

        memory = self.encoder(source)
        output = self.decoder(target, memory)

        output = self.linear(output.mean(dim=0))

        return output
