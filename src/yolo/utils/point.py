import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance(self, point):
        begin = [self.x, self.y]
        end = [point.x, point.y]
        return math.dist(begin, end)
