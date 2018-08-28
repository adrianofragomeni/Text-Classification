import os
import string
import pandas as pd
import numpy as np
import transform_image as ti
import define_info as di
    

original_csv=df = pd.DataFrame(columns=['Font','Character','Bold','Italics'])
c=1
original_npy=[]
# Create Image of all fonts
for font in os.listdir(r'../data/fonts/.'):
    if font=='desktop.ini':
        continue
    dict_info={}
    inf_font=di.text_format(font)
# Create Image of all characters
    for ch in string.printable[:-6]:
        img=ti.create_img(ch,font)
        if img==None:
            continue
        original_npy.append(np.array(img).reshape(64,64,1))
# Upload information on images in info dictionary
        dict_info['Font']=di.clean_font(font)
        dict_info['Character']=ch
        dict_info['Bold']=inf_font[0]
        dict_info['Italics']=inf_font[1]
        original_csv = original_csv.append(dict_info, ignore_index=True)
        c+=1
        
# Save Info in a csv file
original_csv.to_csv('../data/Original_dataset.csv')
# Save Npy file containing the arrays of original images
np.save('../data/all_pics.npy',original_npy)


