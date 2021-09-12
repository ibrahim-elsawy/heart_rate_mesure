import imquality.brisque as brisque
import PIL.Image


def getQuality(dir):
    '''
    check the quality of image return true if good quality
    Args:
        dir (str): directory of the image
    Outputs:
        quality (bool): true if quality of tongue image is good 
    '''

    img = PIL.Image.open(dir)
    qualityScore = brisque.score(img)
    return True if qualityScore < 8 else False


