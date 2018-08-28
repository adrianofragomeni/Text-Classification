# -*- coding: utf-8 -*-
from keras.models import model_from_json
from keras.models import Model
import numpy as np
import transform_image as ti



# load the data
final_npy = np.load("../data/final.npy")

# load the model
with open("../data_out/model/model.json","r") as file:
  loaded_model_json = file.read()

model=model_from_json(loaded_model_json)
model.load_weights("../data_out/model/model.h5")


''' define submodel for each class '''
model_font = Model(model.input, model.get_layer("output_font").output)
model_char = Model(model.input, model.get_layer("output_char").output)
model_bold = Model(model.input, model.get_layer("output_bold").output)
model_ital = Model(model.input, model.get_layer("output_italics").output)


''' plot and save images of the intermediate outputs '''
ti.intermediate_output(final_npy[5], model_font, "font")
ti.intermediate_output(final_npy[5], model_char, "character")
ti.intermediate_output(final_npy[5], model_bold, "bold")
ti.intermediate_output(final_npy[5], model_ital, "italics")


''' plot and save images of the convolution layers '''
ti.nn_layer_image(model, 2, final_npy[5], "first_convolution")
ti.nn_layer_image(model, 4, final_npy[5], "second_convolution")
ti.nn_layer_image(model, 6, final_npy[5], "third_convolution")


