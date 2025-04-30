import time
import json
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

parser = argparse.ArgumentParser(description='Streams a folder to a Spark Streaming Context')
parser.add_argument('--folder', '-f', help='Data folder', required=True, type=str)
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream', required=False, type=bool, default=False)
parser.add_argument('--sleep', '-t', help='Streaming interval', required=False, type=int, default=3)

TCP_IP = "localhost"
TCP_PORT = 6100

LABEL_MAP = {
    'bus': 0,
    'car': 1,
    'truck': 2,
    'motorcycle': 3
}

class Dataset:
    def __init__(self) -> None:
        self.data = []

    def load_images_from_folder(self, folder: str, batch_size: int):
        all_batches = []
        for class_name, label in LABEL_MAP.items():
            class_folder = os.path.join(folder, class_name)
            image_files = [os.path.join(class_folder, img) for img in os.listdir(class_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                images, labels = [], []
                for img_path in batch_files:
                    try:
                        img = Image.open(img_path).convert("RGB").resize((32, 32))
                        img = np.asarray(img).flatten()
                        images.append(img)
                        labels.append(label)
                    except Exception as e:
                        print(f"Failed to process {img_path}: {e}")
                if images:
                    all_batches.append([images, labels])
        return all_batches

    def sendBatchesToSpark(self, tcp_connection, folder, batch_size):
        batches = self.load_images_from_folder(folder, batch_size)
        pbar = tqdm(total=len(batches))
        for batch_id, batch in enumerate(batches):
            images, labels = batch
            payload = {}
            for idx, (img, label) in enumerate(zip(images, labels)):
                payload[idx] = {f'feature-{i}': int(val) for i, val in enumerate(img)}
                payload[idx]['label'] = label

            try:
                tcp_connection.send((json.dumps(payload) + "\n").encode())
                time.sleep(0.1)
            except BrokenPipeError:
                print("Broken pipe error: connection closed or batch too large.")
            except Exception as e:
                print(f"Error: {e}")
            pbar.update(1)
            pbar.set_description(f"Sent batch {batch_id+1}")
            time.sleep(sleep_time)

    def connectTCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")

        print("1) bind/listen ready, about to accept()")
        connection, address = s.accept()
        print("2) accept() unblocked, connected to", address)

        time.sleep(1)
        print(f"Connected to {address}")
        return connection, address

if __name__ == '__main__':
    args = parser.parse_args()

    data_folder = args.folder
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()

    if endless:
        while True:
            dataset.sendBatchesToSpark(tcp_connection, data_folder, batch_size)
    else:
        dataset.sendBatchesToSpark(tcp_connection, data_folder, batch_size)

    tcp_connection.close()
