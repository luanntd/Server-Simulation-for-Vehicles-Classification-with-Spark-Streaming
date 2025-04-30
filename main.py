from modules.trainer import SparkConfig, Trainer
from models import ResnetClassifier
from modules.utils import Transforms, RandomHorizontalFlip, Normalize
import warnings

warnings.filterwarnings("ignore")

transforms = Transforms([
    RandomHorizontalFlip(p=0.345), 
    Normalize(
        mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), 
        std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)
    )
])

if __name__ == "__main__":
    spark_config = SparkConfig

    resnet = ResnetClassifier(lr=0.01)
    trainer = Trainer(resnet, "train", spark_config, transforms)

    trainer.run()
