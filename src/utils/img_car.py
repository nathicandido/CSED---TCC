class ImgCar:
    def __init__(self, b_box, position):
        self.b_box = b_box
        self.position = position

    def get_position(self):
        return self.position

    def get_image(self):
        return self.b_box.image

    def get_b_box_center(self):
        return self.b_box.center

    def __str__(self):
        return f'position: {self.position}, {self.b_box}'
