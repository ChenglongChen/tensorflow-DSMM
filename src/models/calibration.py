

# https://www.kaggle.com/c/quora-question-pairs/discussion/31179
def f(x, a, b):
    return a * x / (a * x + b * (1. - x))


def convert(y_pred, p_online, p_offline):
    a = p_online / p_offline
    b = (1. - p_online) / (1. - p_offline)
    return f(y_pred, a, b)


if "__name__" == "__main__":
    x = 0.50296075348400959
    p_online = 0.50296075348400959
    p_offline = 0.5191087559849992
    print(convert(x, p_online, p_offline))
