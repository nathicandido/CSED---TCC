class JSONDatasetBuilder:

    @classmethod
    def build_json_for_training(cls, array_x, array_y, label):
        data = list()
        for x_matrix, y_matrix in zip(array_x, array_y):
            for x_sig, y_sig in zip(x_matrix, y_matrix):
                data.append(
                    dict(
                        x_sig=x_sig,
                        y_sig=y_sig,
                        label=label
                    )
                )

        return data
