class Size:
    def __init__(self, length, height):
        self.length = length
        self.height = height

    def __str__(self):
        return f'({self.length}, {self.height})'
