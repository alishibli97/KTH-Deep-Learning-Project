import argparse

class config:
    def __init__(self):
        self.data_dir = None
        self.classes = None
        self.lr = None
        self.epochs = None
        self.batch = None
        self.iter = None

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', required=True, type=str)
        parser.add_argument('--classes', required=True, type=int)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch', type=int, default=10)
        parser.add_argument('--iter', type=int, default=10)
        
        cfg = config()
        args = parser.parse_args()
        cfg.data_dir = args.data_dir
        cfg.classes = args.classes
        cfg.lr = args.learning_rate
        cfg.epochs = args.epochs
        cfg.batch = args.batch
        cfg.iter = args.iter

        return cfg
