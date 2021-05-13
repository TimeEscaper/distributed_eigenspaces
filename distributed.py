#!/usr/bin/env python

import pika
import json
import argparse
import time
import numpy as np

from load_data import load_CIFAR_10_data
from scipy.linalg import eigh as largest_eigh


class Node:
    def __init__(self, broker_host):
        # TODO: Implement proper credentials management
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=broker_host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='master')
        self.channel.queue_declare(queue='slaves')

    def top_k_eigenvectors(self, matrix, k):
        """
        params: matrix: matrix
                k: number of top vector to extract
        return: top k eignvectors based on eignvalues
        """
        N = matrix.shape[0]
        return largest_eigh(matrix, eigvals=(N - k, N - 1))[1]


class SlaveNode(Node):
    def __init__(self, broker_host, data):
        super().__init__(broker_host)
        print("Slave Start listening")
        self.data = data
        self.channel.basic_consume(queue='slaves', on_message_callback=self.callback_)

    def start(self):
        self.channel.start_consuming()

    def callback_(self, channel, method, properties, body):
        request = json.loads(body)
        print("Slave: Received, batchid: " + str(request["batch"]))

        batch = self.data[request["batch"][0]:request["batch"][1]]
        eigenspace = self.compute_sigma_hat_(batch)
        eigenspace = self.top_k_eigenvectors(eigenspace, request["rank"])
        response = dict()
        response["batch"] = request["batch"]
        response["eigenspace"] = eigenspace.tolist()
        self.send_to_master_(str(json.dumps(response)))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_master_(self, message):
        print("Sending to Master")
        self.channel.basic_publish(exchange='', routing_key='master', body=message)

    def compute_sigma_hat_(self, x):
        """
        Compute the segma hat that can run nodes(slaves)
        params: x: batch of the data
        return: segma hat
        """
        # the K leading eigenvectors of simga = 1/n * sum{X @ X.T} over n
        n, d = x.shape
        sigma_hat = np.zeros((d, d))
        sigma_hat += np.dot(x.T, x)
        sigma_hat /= n
        return sigma_hat

    def compute_eigenspace_(self, batch, rank):
        sigma = np.zeros((len(batch[0]), len(batch[0])))
        for x in batch:
            vec = np.array(x)
            sigma += vec @ vec.T
        sigma = sigma / len(batch)
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        return eigenvectors[:, 0:rank]


class MasterNode(Node):
    def __init__(self, broker_host, rank, batches_number, data):
        super().__init__(broker_host)
        self.rank = rank
        self.batches_number = batches_number
        self.data = data
        self.batches_in_process = set()
        self.batches = list()
        self.computed_eigens = list()
        self.current_batch = 0
        print("Master Start listening")
        self.start_time = time.time()
        self.channel.basic_consume(queue='master', on_message_callback=self.callback_)

    def start(self):
        # Split the dataset
        print("Splitting dataset...")
        step = self.data.shape[0] // self.batches_number
        for i in range(self.batches_number):
            batch = (i * step, (i + 1) * step)
            self.batches.append(batch)
            #self.batches.append(self.data[i*step:(i+1)*step])
            self.batches_in_process.add(batch)

        # Send batches to queue
        print("Sending to slaves...")
        for i in range(0, 5):
            request = dict()
            request["rank"] = self.rank
            request["batch"] = self.batches.pop()
            self.send_to_slaves_(str(json.dumps(request)))

        print("Start waiting for messages")
        self.channel.start_consuming()

    def callback_(self, channel, method, properties, body):
        request = json.loads(body)
        batch = (request["batch"][0], request["batch"][1])
        print("Master: Received, batch: " + str(batch))

        eigenspace = np.array(request["eigenspace"])
        self.computed_eigens.append(eigenspace)
        self.batches_in_process.remove(batch)

        if len(self.batches_in_process) == 0:
            sigma_tilde = np.zeros((self.computed_eigens[0].shape[0], self.computed_eigens[0].shape[0]))
            for eigen in self.computed_eigens:
                sigma_tilde += eigen @ eigen.T
            sigma_tilde /= self.batches_number
            print("Computed! Time (seconds): " + str(time.time() - self.start_time))
        elif len(self.batches) != 0:
            request = dict()
            request["batch"] = self.batches.pop()
            request["rank"] = self.rank
            self.send_to_slaves_(str(json.dumps(request)))
            print("Sended batch: " + str(request["batch"]))

        channel.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_slaves_(self, message):
        print("Sending to slaves: ")
        self.channel.basic_publish(exchange='', routing_key='slaves', body=message)


def run_master(broker, rank, batches_number, data):
    master = MasterNode(broker, rank, batches_number, data)
    master.start()


def run_slave(broker, data):
    slave = SlaveNode(broker, data)
    slave.start()


def main():
    parser = argparse.ArgumentParser(description="Multinode PCA")
    parser.add_argument("--mode", help="Mode to run script - slave or master")
    parser.add_argument("--broker", help="Message broker IP address")
    parser.add_argument("--rank", help="Approximation rank (only for master node")
    parser.add_argument("--batches", help="Total batches number")
    parser.add_argument("--data", default="cifar-10-batches-py", help="Path to dataset")

    args = parser.parse_args()

    if args.broker is None:
        raise RuntimeError("Broker not specified")

    data, filenames, labels = load_CIFAR_10_data(args.data)
    # Remove RGB Channales and make dataset single scale
    data = data.mean(axis=3)
    # Reshape images to be in R(1024) instead of R(32x32)
    data = data.reshape(data.shape[:-2] + (-1,))

    if args.mode == "slave":
        run_slave(args.broker, data)
    elif args.mode == "master":
        run_master(args.broker, int(args.rank), int(args.batches), data)
    else:
        raise RuntimeError("Mode not specified or specified wrong")


if __name__ == "__main__":
    main()
