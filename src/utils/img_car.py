class ImgCar:
    def __init__(self, b_box, position):
        self.b_box = b_box
        self.position = position

    def get_position(self):
        return self.position

    def get_image(self):
        return self.b_box.image
