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

        result = self.workers[0].compute_frobenius_norm(matrix).value
        frobenius_norm = math.sqrt(result)

        total_time = time.time() - start_time
        self.write_output(original_matrix, frobenius_norm, total_time, n)
        print("Job Finished")

    @staticmethod
    @expose
    def compute_frobenius_norm(matrix):
        total_sum = 0.0
        for row in matrix:
            for element in row:
                total_sum += element ** 2
        return total_sum

    def read_input(self):
        with open(self.input_file_name, 'r') as f:
            n = int(f.readline().strip())
        return n

    def write_output(self, matrix, frobenius_norm, total_time, n):
        with open(self.output_file_name, 'w') as f:
            f.write("\nSequential Frobenius Norm Calculation\n")
            f.write("Matrix dimension: {} x {}\n".format(n, n))
            f.write("Total computation time: {:.6f} seconds\n\n".format(total_time))
            f.write("\nFrobenius Norm: {:.6f}\n".format(frobenius_norm))