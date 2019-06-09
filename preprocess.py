# IMPORT
import os
import librosa
import librosa.display
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image

def convertAudioToSpectrogram(pathAudio, pathImg, resize, size, color):
    '''
    Permet de convertir l'ensemble des données audio de notre dataset
    en spectrograme
    :param pathAudio: chemin vers le dossier contenant le dataset sous format audio
    :param pathImg: chemin qui va contenir nos spectrogramme de nos fichiers audios
    :param ifResize: boolean permettant de resize les images
    :param size: taille des images si resize
    :color: définit l'enrtegistrement des images en noir/blanc ou couleur
    '''
    FINAL_SIZE = None
    FINAL_COLOR = None
    if (color):
        FINAL_COLOR = "RGB"
    else:
        FINAL_COLOR = "L"


    for musicClass in os.listdir(pathAudio):
        pathCurrentClass = pathAudio + "/" + musicClass
        pathCurrentMusicForSavingDirectory = pathImg + "/" + musicClass

        if not os.path.exists(pathCurrentMusicForSavingDirectory):
            os.makedirs(pathCurrentMusicForSavingDirectory)

        for musicFile in tqdm(os.listdir(pathCurrentClass), "Conversion de la classe : {}".format(musicClass)):
            pathCurrentMusic = pathCurrentClass + "/" + musicFile
            pathCurrentMusicForSavingFile = pathCurrentMusicForSavingDirectory + "/" + str(musicFile).replace(".au","")+ ".png"
            y, sr = librosa.load(pathCurrentMusic)
            #, fmax=9000, n_fft=1024
            spectro = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, fmax=9000)

            #pylab.axis('off')  # no axis
            #pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge

            librosa.display.specshow(librosa.power_to_db(spectro, ref=np.max))
            #librosa.display.specshow(librosa.amplitude_to_db(spectro, ref=np.max))

            plt.axis('off')
            plt.tight_layout(pad=0)


             #plt.show()

            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()


            if(resize):
                FINAL_SIZE = size
            else:
                FINAL_SIZE = canvas.get_width_height()





            img = Image.frombytes(FINAL_COLOR, FINAL_SIZE, canvas.tostring_rgb())
            img.save(pathCurrentMusicForSavingFile)
            plt.close()

            #pylab.savefig(pathCurrentMusicForSavingFile, bbox_inches=None, pad_inches=0)
            #pylab.close()



if __name__ == '__main__':
    pathAudioFolder = "./rawData"
    pathImgFolder = "./imgData"
    resizeImg = False
    imgSize = (5, 5) # in inch
    imgColor = True


    convertAudioToSpectrogram(pathAudio=pathAudioFolder, pathImg=pathImgFolder, resize=resizeImg, size=imgSize, color=imgColor)