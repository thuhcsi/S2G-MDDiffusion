from args import parse_train_opt
from LMDM import LMDM


def train(opt):
    model = LMDM(opt.feature_type, opt.checkpoint)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
