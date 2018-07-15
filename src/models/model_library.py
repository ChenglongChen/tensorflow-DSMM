
from models.bcnn import BCNN, ABCNN1, ABCNN2, ABCNN3
from models.decatt import DecAtt
from models.dssm import DSSM, CDSSM, RDSSM
from models.dsmm import DSMM
from models.esim import ESIM
from models.match_pyramid import MatchPyramid, GMatchPyramid


def get_model(model_type):
    if model_type == "dssm":
        return DSSM
    elif model_type == "cdssm":
        return CDSSM
    elif model_type == "rdssm":
        return RDSSM
    elif model_type == "match_pyramid":
        return MatchPyramid
    elif model_type == "g_match_pyramid":
        return GMatchPyramid
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
    elif model_type == "esim":
        return ESIM
    elif model_type == "decatt":
        return DecAtt
    else:
        return DSMM
