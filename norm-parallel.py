from __future__ import print_function, division
from Pyro4 import expose
import random
import time
import math


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")

    @staticmethod
    def generate_random_matrix(size, value_range=(1, 10)):
        matrix = []
        for _ in range(size):
            row = [random.randint(*value_range) for _ in range(size)]
            matrix.append(row)
        return matrix

    def solve(self):
        print("Job Started")
        n = self.read_input()
        matrix = self.generate_random_matrix(n)

        original_matrix = [row[:] for row in matrix]

        start_time = time.time()

        flat_matrix = []
        for row in matrix:
            flat_matrix.extend(row)

        num_workers = len(self.workers)
        elements_per_worker = len(flat_matrix) // num_workers

        mapped = []
        for i in range(num_workers):
            start_idx = i * elements_per_worker
            end_idx = start_idx + elements_per_worker if i < num_workers - 1 else len(flat_matrix)

            if start_idx >= len(flat_matrix):
                break

            chunk = flat_matrix[start_idx:end_idx]
            mapped.append(self.workers[i].compute_partial_sum(chunk))

        partial_sums = [result.value for result in mapped]
        total_sum = sum(partial_sums)

        frobenius_norm = math.sqrt(total_sum)

        total_time = time.time() - start_time

        self.write_output(original_matrix, frobenius_norm, total_time, n, len(mapped))
        print("Job Finished")

    @staticmethod
    @expose
    def compute_partial_sum(chunk):
        return sum(x * x for x in chunk)

    def read_input(self):
        with open(self.input_file_name, 'r') as f:
            n = int(f.readline().strip())
        return n

    def write_output(self, matrix, frobenius_norm, total_time, n, num_workers):
        with open(self.output_file_name, 'w') as f:
            f.write("\nParallel Frobenius Norm Calculation\n")
            f.write("Matrix dimension: {} x {}\n".format(n, n))
            f.write("Number of workers: {}\n".format(num_workers))
            f.write("Total computation time: {:.6f} seconds\n\n".format(total_time))
            f.write("\nFrobenius Norm: {:.6f}\n".format(frobenius_norm))