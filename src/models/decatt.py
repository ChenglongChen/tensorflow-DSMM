
from copy import copy

from models.esim import ESIMDecAttBaseModel


class DecAtt(ESIMDecAttBaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        p = copy(params)
        # model config
        p.update({
            "model_name": p["model_name"] + "dec_att",
            "encode_method": "project",
            "attend_method": ["ave", "max", "min", "self-attention"],

            "project_type": "fc",
            "project_hidden_units": [64 * 4, 64 * 2, 64],
            "project_dropouts": [0, 0, 0],

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [64 * 4, 64 * 2, 64],
            "fc_dropouts": [0, 0, 0],
        })
        super(DecAtt, self).__init__(p, logger, init_embedding_matrix)
