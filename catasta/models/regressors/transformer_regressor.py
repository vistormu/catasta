from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Linear

from .regressor import Regressor


class TransformerRegressor(Regressor):
    def __init__(self, *,
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

        self.linear = Linear(in_features=d_model,
                             out_features=1,
                             bias=bias,
                             )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        memory = self.encoder(source)
        output = self.decoder(target, memory)

        output = self.linear(output.mean(dim=0))

        return output
