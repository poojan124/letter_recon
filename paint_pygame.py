import sys,pygame
from pygame.locals import *
import image_to_mnist
import test
import cv2 as cv
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
from PIL import Image

pygame.init()

screen = pygame.display.set_mode((500,500))
screen.fill((255,255,255))

brush = pygame.image.load("brush.png")
brush = pygame.transform.scale(brush,(15,15))

pygame.display.update()

clock = pygame.time.Clock()

z = 0
while(1):
    clock.tick(60)
    x,y = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            z = 1
        elif event.type == MOUSEBUTTONUP:
            z = 0
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill((255,255,255))
                pygame.display.update()
            elif event.key == pygame.K_t:
                pygame.image.save(screen,"image.png")
                img = image_to_mnist.convert("image.png")
                #----------mapping block----------#
                
                mapping = [0]*47
                f_ = open("emnist-balanced-mapping.txt",'r')
                for line in f_:
                    a,b=line.split()
                    mapping[int(a)]=chr(int(b))
                f_.close()
                
                #its started
                json_file = open('model.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights("model.h5")
                
                #------------------------------#
                test_im = Image.open("mnist.png")
                test_im_arr = np.array(test_im)
                test_im_arr = np.transpose(test_im_arr)
                temp_im_ = Image.fromarray(test_im_arr)
                test_im_arr = test_im_arr.astype('float')/255.0
                test_im_arr = test_im_arr.reshape((28*28))
                test_im_arr = np.atleast_2d(test_im_arr)
                #mapping[np.array(model.predict([test_im_arr])[0]).argmax(axis=0)]
                opp = loaded_model.predict(test_im_arr)
                #print(np.array(opp[0]).argmax(axis=0))
                print("And The character is : -> ",mapping[np.array(opp[0]).argmax(axis=0)],"\n\n")


                #cv.imshow('img-windows',img)
                #cv.waitKey(0)
                #cv.imwrite('OP.png',img)
        if z == 1:
            screen.blit(brush,(x-2,y-2))
            pygame.display.update()
