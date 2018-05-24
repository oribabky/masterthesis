def faceRecognizer(Image img):
    for each pixel in img.pixel:
        if faceRGB(pixel) == False
            continue
        if pixel.previousPixel.rgb == pixel.rgb
            or pixel.previousPixel.previousPixel.rgb == pixel.rgb
            or ...


def faceRecognizerML([Image] faces, Image img):
    Model model
    for each face in faces:
        model.learn(face)

    return model.classifyFace(img)