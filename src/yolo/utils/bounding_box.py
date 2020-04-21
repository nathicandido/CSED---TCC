
class BoundingBox:
    def __init__(self, center, size, image, score):
        self.center = center
        self.size = size
        self.image = image
        self.score = score
