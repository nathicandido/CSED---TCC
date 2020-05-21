class BoundingBox:
    def __init__(self, center, size, image, score):
        self.center = center
        self.size = size
        self.image = image
        self.score = score

    def __str__(self):
        return f'center: {self.center}, size: {self.size}, score: {self.score}'
