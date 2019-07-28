from PIL import Image

i1 = Image.open("../images/999900.jpg")
i2 = Image.open("../images/9999300.jpg")
i3 = Image.open("../images/9999400.jpg")

h1 = i1.histogram()
h2 = i2.histogram()
h3 = i3.histogram()

# Then, one may use the np.histogram function and funcstion about histogram intersection to intersection histograms.
