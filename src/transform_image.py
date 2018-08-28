import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_dilation,grey_erosion
from scipy.ndimage.interpolation import spline_filter
from numpy.random import choice
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
import keras.backend as K



''' creation and transformation of images '''


def create_img(ch,font):
    ''' create 64x64 images of each character '''
    img = Image.new(mode='L',size=(64, 64),color=255)
    d = ImageDraw.Draw(img)
    fnt=ImageFont.truetype('../data/fonts/'+font, size=32)
    w, h = d.textsize(ch, font=fnt)
    # Centre the image
    d.text(((64-w)/2,(64-h)/2),ch,fill=0, font=fnt)
    img=img.resize((64,64),Image.LANCZOS)
    if not ImageChops.invert(img).getbbox():
        return None
    else:
        return(img)


def add_line(pic):
    ''' add random vertical/horizontal/diagonal lines to the picture '''
    val=random.randint(1,3)
    if val==1:
        np.fill_diagonal(pic.reshape(64,64), 0)
    elif val==2:
        factor = random.randint(1,63)
        pic.reshape(64,64)[factor:(factor+1), 0:64] = 0
    elif val==3:
        factor = random.randint(1,63)
        pic.reshape(64,64)[0:64, factor:(factor+1)] = 0
    return pic


def add_filter(pic):
    ''' add random filter to the picture '''
    val=val=random.randint(1,2)
    if val==1:
        s=random.uniform(0.8,1.2)
        pic=gaussian_filter(pic,sigma=s)
    elif val==2:
        pic=spline_filter(pic)
    return pic


def ruined_pic(pic):
    ''' alter the number of pixels in the image: increase or decrease character width '''
    val=random.randint(1,2)
    if val==1:
        pic=grey_dilation(pic.reshape(64,64), size=(2,1))
    else:
        pic=grey_erosion(pic.reshape(64,64), size=(2,1))
    return pic


def modify(pic):
    ''' randomly pick the permutations defined above (addition of lines, filters and pixels) '''
    if bool(choice([True,False],1, p=[0.4,0.6])):
        pic=add_line(pic)
    if bool(choice([True,False],1, p=[0.4,0.6])):
        pic=add_filter(pic)
    if bool(choice([True,False],1, p=[0.4,0.6])):
        pic=ruined_pic(pic)
    return(pic.reshape(64,64,1))


def intermediate_output(image, label, filename):
    ''' takes as input an image as a numpy array and the label (font, character,
    bold or italics), and returns the image of the related (intermediate) layer output'''
    # reshape the input image from (64,64,1) to (1,64,64,1) in order to use the predict function
    input_image = image.reshape((1,) + image.shape)
    im = label.predict(input_image)
    # save the image as .png file to the specified folder
    pic = Image.fromarray((im * 255).astype(np.uint8))
    pic.resize((pic.size[0]*64, 64))
    pic.save("../data_out/img/" + filename + ".png", "PNG")


def nn_layer_image(model_, layer, image, filename):
    ''' takes as input the model, layer and an image as a numpy array and returns
    a visual representation of the output of the chosen layer (e.g. convolution)'''
    # define outputs of the chosen layer
    output_layer = model_.layers[layer].output
    # define a K function that returns the output of the chosen layer for an input image
    output_fn = K.function([model_.layers[0].input], [output_layer])
    # reshape the input image from (64,64,1) to (1,64,64,1)
    input_image = image.reshape((1,) + image.shape)
    # obtain output associated to the chosen layer for the input image
    output_image = output_fn([input_image])
    list_array = np.asarray(output_image)
    output_image1 = list_array[0,:]
    # change the shape of the output in order to plot it and save it as image
    output_image2 = np.rollaxis(np.rollaxis(output_image1,3,2),3,2)
    pic = Image.fromarray((output_image2[0,:,:,6] * 255).astype(np.uint8)).resize((64,64))
    pic.save("../data_out/img/" + filename + ".png", "PNG")
