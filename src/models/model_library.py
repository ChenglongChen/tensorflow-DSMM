
from models.dssm import DSSM, CDSSM
from models.dsmm import DSMM
from models.match_pyramid import MatchPyramid


def get_model(model_type):
    if model_type == "dssm":
        return DSSM
    elif model_type == "cdssm":
        return CDSSM
    elif model_type == "match_pyramid":
        return MatchPyramid
    elif model_type == "dsmm":
        return DSMM
    else:
        return DSMM
