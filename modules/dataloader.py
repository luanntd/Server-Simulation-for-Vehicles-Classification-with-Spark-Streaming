import json
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector

from .utils import Transforms
from .trainer import SparkConfig

class DataLoader:
    def __init__(self, 
                 sparkContext:SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig,
                 transforms: Transforms) -> None:
        
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )
        self.transforms = transforms

    def parse_stream(self) -> DStream:
        transforms = self.transforms

        def safe_parse(line):
            try:
                data = json.loads(line)
                return data.values()
            except Exception as e:
                print(f"[DataLoader] Failed to parse line: {e}")
                return []

        def safe_unpack(x):
            try:
                values = list(x.values())
                image = np.array(values[:-1]).reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
                label = values[-1]
                return [image, label]
            except Exception as e:
                print(f"[DataLoader] Failed to unpack data: {e}")
                return None

        json_stream = self.stream.flatMap(safe_parse)
        parsed = json_stream.map(safe_unpack).filter(lambda x: x is not None)
        return DataLoader.preprocess(parsed, transforms)
    
    @staticmethod
    def preprocess(stream: DStream, transforms: Transforms) -> DStream:
        stream = stream.map(lambda x: [transforms.transform(x[0]).reshape(32, 32, 3).reshape(-1).tolist(),x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
        
        return stream
