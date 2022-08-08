import cv2
from cv2 import resize
import numpy as np
import magic


def faceClassifier(image):
    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    my_type = (magic.from_file(image, mime=True))

    if 'image' in my_type:

        image = cv2.imread(image)
        
        img_resized = resizing_img(image)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceClassif.detectMultiScale(img_resized,
            scaleFactor = 1.2,   #tamaño de la imagen, hace una piramide de imagenes (con valores pequeños, aumentan falsos positivos)
            minNeighbors=5,     #cuantos vecinos tienen los rectangulos candidatos para retener
            minSize=(30,30),     #tamaño minimo del rostro
            maxSize=(400,400)    #tamaño maximo del rostro
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img_resized,(x,y), (x+w,y+h),(0,255,0),2)

        #cv2.imshow('imagegray',gray)
        cv2.imshow('image', img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif 'video' in my_type:            #don't put a high duration video, processing time will be long
        print(image)
        video = cv2.VideoCapture('video.mp4')
        while True:
            ret, frame = video.read()
            resized_frame = resizing_img(frame)
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            gray = (gray * 255).astype(np.uint8)

            faces = faceClassif.detectMultiScale(resized_frame,
                scaleFactor = 1.2,   #tamaño de la imagen, hace una piramide de imagenes (con valores pequeños, aumentan falsos positivos)
                minNeighbors=5,     #cuantos vecinos tienen los rectangulos candidatos para retener
                minSize=(30,30),     #tamaño minimo del rostro
                maxSize=(400,400)    #tamaño maximo del rostro
            )
            for(x,y,w,h) in faces:
                cv2.rectangle(resized_frame,(x,y), (x+w,y+h),(0,255,0),2)
            cv2.imshow('frame',resized_frame)

            k = cv2.waitKey(30) & 0xFF   #press ESC to exit 
            if k ==27:
                break
        video.release()
        cv2.destroyAllWindows()

    else:
        print('not supported typefile')
    
def resizing_img(image):
    img_resized = image
    if image.shape[1] > 1000 and image.shape[0] > 500:
        img_resized = cv2.resize(image,(int(image.shape[1]/1.4),int(image.shape[0]/1.4)),interpolation = cv2.INTER_AREA)
    if image.shape[1] > 1500 and image.shape[0] > 700:
        img_resized = cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)),interpolation = cv2.INTER_AREA)
    if image.shape[1] > 2000 and image.shape[0] > 1500:
        img_resized = cv2.resize(image,(int(image.shape[1]/3),int(image.shape[0]/3)),interpolation = cv2.INTER_AREA)
    if image.shape[1] > 3000 and image.shape[0] > 2000:
        img_resized = cv2.resize(image,(int(image.shape[1]/4.3),int(image.shape[0]/4.3)),interpolation = cv2.INTER_AREA)
    if image.shape[1] > 4000 or image.shape[0] > 3000:
        img_resized = cv2.resize(image,(int(image.shape[1]/5.5),int(image.shape[0]/5.5)),interpolation = cv2.INTER_AREA)

    if image.shape[1] < 200 and image.shape[0] < 150:
        img_resized = cv2.resize(image,(int(image.shape[1]*3),int(image.shape[0]*3)),interpolation = cv2.INTER_AREA)
    if image.shape[1] < 400 and image.shape[0] < 300:
        img_resized = cv2.resize(image,(int(image.shape[1]*2.5),int(image.shape[0]*2.5)),interpolation = cv2.INTER_AREA)  
    if image.shape[1] < 600 and image.shape[0] < 500:
        img_resized = cv2.resize(image, (int(image.shape[1]*1.3) , int(image.shape[0]*1.3)), interpolation = cv2.INTER_AREA)
    
    return img_resized