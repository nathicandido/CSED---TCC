from src.utils.distance_calculator import DistanceCalculator


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance(self, point):
        begin = [self.x, self.y]
        end = [point.x, point.y]
        return DistanceCalculator.n_dim_euclidean_distance(begin, end)

    def __str__(self):
        return f'({self.x}, {self.y})'
