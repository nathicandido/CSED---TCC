from typing import List

from src.yolo.utils import bounding_box
from src.yolo.utils.img_carro import ImgCarro


class Frame:
    carros = [ImgCarro]

    def __init__(self, carros: List[ImgCarro]):
        # self.carros = self.deletar_duplicados(carros)
        self.carros = carros

    # def deletar_duplicados(self, carros: List[ImgCarro]):
        # for carro em carros
            # if A.posicao === B.posicao
            #     delete B
        # return carros
