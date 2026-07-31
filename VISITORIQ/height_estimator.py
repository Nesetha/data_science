class HeightEstimator:

    def estimate(self, pixel_height):

        estimated_cm = int(pixel_height * 0.45)

        return estimated_cm