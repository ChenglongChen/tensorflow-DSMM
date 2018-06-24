
from models.bcnn import BCNN, ABCNN1, ABCNN2, ABCNN3
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
    elif model_type == "bcnn":
        return BCNN
    elif model_type == "abcnn1":
        return ABCNN1
    elif model_type == "abcnn2":
        return ABCNN2
    elif model_type == "abcnn3":
        return ABCNN3
    else:
        return DSMM
