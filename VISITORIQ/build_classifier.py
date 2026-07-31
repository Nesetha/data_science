class BuildClassifier:

    def classify(self, width, height):

        if height == 0:
            return "Unknown"

        ratio = width / height

        if ratio < 0.45:
            return "Slim"

        elif ratio < 0.65:
            return "Average"

        else:
            return "Broad"