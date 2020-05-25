class JSONDatasetBuilder:

    @classmethod
    def build_json_from_patterns(cls, patterns):
        data = list()
        for pattern in patterns:
            try:
                for cp in pattern:
                    data.append(
                        dict(
                            x_sig=cp.x_sig,
                            y_sig=cp.y_sig,
                            label=cp.label
                        )
                    )
            except TypeError:  # Getting original classed patterns
                data.append(
                    dict(
                        x_sig=pattern.x_sig,
                        y_sig=pattern.y_sig,
                        label=pattern.label
                    )
                )

        return data
