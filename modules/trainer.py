import os
import pyspark
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT

from .utils import Transforms
from .dataloader import DataLoader

class SparkConfig:
    appName = "Vehicles_Classification"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 3
    model_checkpoint_path = "checkpoints/model.pkl"


class Trainer:
    def __init__(self, model, split: str, spark_config: SparkConfig, transforms: Transforms):
        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf, self.transforms)

        if self.split == "test":
            if os.path.exists(self.sparkConf.model_checkpoint_path):
                print(f"[INFO] Loading model from checkpoint: {self.sparkConf.model_checkpoint_path}")
                self.model.load(self.sparkConf.model_checkpoint_path)
            else:
                raise FileNotFoundError(f"Checkpoint not found at {self.sparkConf.model_checkpoint_path}")

        # Tracking for metrics
        self.batch_count = 0
        self.test_accuracy = 0
        self.test_loss = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_f1 = 0
        self.cm = None


    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)

        try:
            self.ssc.start()
            self.ssc.awaitTermination()
        except KeyboardInterrupt:
            print("Stopping streaming context...")
            self.ssc.stop(stopSparkContext=True, stopGraceFully=True)
            print("Streaming context stopped.")


    def __train__(self, timestamp, rdd: pyspark.RDD):
        try:
            print(f"[Spark] New RDD at {timestamp}")
            if not rdd.isEmpty():
                print("Total Batch Size of RDD Received :", rdd.count())

                schema = StructType([
                    StructField("image", VectorUDT(), True),
                    StructField("label", IntegerType(), True)
                ])
                
                df = self.sqlContext.createDataFrame(rdd, schema)
                predictions, accuracy, precision, recall, f1 = self.model.train(df)

                print("=" * 10)
                print(f"Predictions = {predictions}")
                print(f"Accuracy = {accuracy}")
                print(f"Precision = {precision}")
                print(f"Recall = {recall}")
                print(f"F1 Score = {f1}")
                print("=" * 10)

                # Save checkpoint
                print(f"[INFO] Saving model to {self.sparkConf.model_checkpoint_path}")
                self.model.save(self.sparkConf.model_checkpoint_path)

            else:
                print("[Spark] Empty RDD")

        except Exception as e:
            print(f"[Spark ERROR] Exception in __train__: {e}")


    def predict(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__predict__)

        try:
            self.ssc.start()
            self.ssc.awaitTermination()
        except KeyboardInterrupt:
            print("Stopping streaming context...")
            self.ssc.stop(stopSparkContext=True, stopGraceFully=True)
            print("Streaming context stopped.")


    def __predict__(self, rdd: pyspark.RDD):
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)
            ])

            df = self.sqlContext.createDataFrame(rdd, schema)

            acc, loss, prec, rec, f1, cm = self.model.predict(df)

            self.batch_count += 1
            self.test_accuracy += acc
            self.test_loss += loss
            self.test_precision += prec
            self.test_recall += rec
            self.test_f1 += f1
            self.cm = cm if self.cm is None else self.cm + cm

            print(f"Batch {self.batch_count}")
            print(f"Avg Accuracy: {self.test_accuracy / self.batch_count:.4f}")
            print(f"Avg Loss: {self.test_loss / self.batch_count:.4f}")
            print(f"Avg Precision: {self.test_precision / self.batch_count:.4f}")
            print(f"Avg Recall: {self.test_recall / self.batch_count:.4f}")
            print(f"Avg F1 Score: {self.test_f1 / self.batch_count:.4f}")
            print("Confusion Matrix:\n", self.cm)

        else:
            print("[Spark] Empty RDD received for prediction.")

    def run(self):
        if self.split == 'train':
            print("[INFO] Starting training...")
            self.train()
        elif self.split == 'test':
            print("[INFO] Starting evaluation...")
            self.predict()
        else:
            raise ValueError(f"[ERROR] Unknown split value: '{self.split}'. Use 'train' or 'test'.")

    def get_avg_metrics(self):
        if self.batch_count == 0:
            return None
        
        avg_accuracy = self.test_accuracy / self.batch_count
        avg_loss = self.test_loss / self.batch_count
        avg_precision = self.test_precision / self.batch_count
        avg_recall = self.test_recall / self.batch_count
        avg_f1 = self.test_f1 / self.batch_count

        return {
            "avg_accuracy": avg_accuracy,
            "avg_loss": avg_loss,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "confusion_matrix": self.cm
        }
