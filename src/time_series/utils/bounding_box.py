class BoundingBox:

    def __init__(self, pos_x, pos_y, label=None):
        self.pos_x = pos_x
        self.pos_y = pos_y
        # self.width = w
        # self.height = h
        # self.score = score
        self.label = label

    def __str__(self):
        return f'<BoundingBox: {self.label}, {self.pos_x}, {self.pos_y}>'
