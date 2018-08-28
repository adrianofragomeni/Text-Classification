from keras.models import model_from_json
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import os
import string
import define_info as di

''' obtain binary values of font and character '''

#list all possible fonts
lst_font = [di.clean_font(font) for font in os.listdir(r'../data/fonts/.')]
lst_font = np.unique(lst_font)

#define label binarizer encoder
lb_font = LabelBinarizer()
dummy_font= lb_font.fit_transform(lst_font)

lb_char = LabelBinarizer()
dummy_char=lb_char.fit_transform(list(string.printable[:-6]))

# Import model
with open("../data_out/model/model.json","r") as file:
  loaded_model_json = file.read()

#Import weights
model=model_from_json(loaded_model_json)
model.load_weights("../data_out/model/model.h5")

data=np.load(sys.argv[1])
name=sys.argv[2]

prediction_csv = pd.DataFrame(columns=['Character','Font','Bold','Italics'])

'''Predict image values for each class (i.e. Font, Character, Bold, Italics)'''
for pic in data:
    prediction=model.predict(pic.reshape(1,64,64,1))
    
    pred_font=di.predict_categorical(lb_font,11,prediction[0])[0]
    pred_char=di.predict_categorical(lb_char,94,prediction[1])[0]
    pred_bold=di.predict_bool(prediction[2])[0]
    pred_ital=di.predict_bool(prediction[3])[0]
    # Upload information on images to info dictionary
    dict_info={}
    dict_info['Character']=pred_char
    dict_info['Font']=pred_font
    dict_info['Bold']=pred_bold
    dict_info['Italics']=pred_ital
    prediction_csv=prediction_csv.append(dict_info, ignore_index=True)
    
prediction_csv.to_csv('../data_out/test/'+name+'.csv',header=None,index=False)

