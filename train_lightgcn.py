from trainers.LightGCNTrainer import LightGCNTrainer
from utils.utils import parse_yaml_config

if __name__ == "__main__":
    config = parse_yaml_config('config/train_LightGCN.yaml')
    print(config)

    trainer = LightGCNTrainer(config)
    trainer.train()
