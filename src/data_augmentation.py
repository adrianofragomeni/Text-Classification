import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import random
import transform_image as ti

''' Augment the dataset by applying certain permutations to each image'''
datagen = ImageDataGenerator(
        # Rotation range from -25 to 25
        rotation_range=random.sample([-25,25],1)[0],
        # character width
        width_shift_range=5.,
        # character heigth 
        height_shift_range=7.,
        # rescale picture
        rescale=1.5,
        # zoom picture
        zoom_range=.5,
        # add filters (see transform_image.py)
        preprocessing_function= ti.modify)


#load npy file
original_npy = np.load('../data/all_pics.npy')
# load original dataset
original_csv = pd.read_csv('../data/Original_dataset.csv')
original_csv.drop(original_csv.columns[0], inplace=True, axis=1)


augmented_data=[]
augmented_csv = pd.DataFrame(columns=['Font','Character','Bold','Italics'])
for num in range(original_npy.shape[0]):
    pic = original_npy[num].reshape((1,) + original_npy[num].shape)
    info_pic=original_csv.loc[num,]
    i = 0
    # create 31 modified images per original picture
    for batch in datagen.flow(pic,batch_size=1,shuffle=False,seed=10):
        augmented_data.append(batch.reshape(64,64,1))
        # Upload information on images to info dictionary
        dict_info={}
        dict_info['Font']=info_pic['Font']
        dict_info['Character']=info_pic['Character']
        dict_info['Bold']=info_pic['Bold']
        dict_info['Italics']=info_pic['Italics']
        augmented_csv = augmented_csv.append(dict_info, ignore_index=True)
        i += 1
        if i > 30:
            break  


# save augmented dataset
np.save('../data/augmented_pics.npy',augmented_data)
augmented_csv.to_csv('../data/augmented_dataset.csv')


# create final dataset (original + augmented)
augmented_npy = np.load('../data/augmented_pics.npy')
final_dataset = original_csv.append(augmented_csv,ignore_index=True)
# rescale values between 0 and 1
final_npy = np.concatenate((original_npy, augmented_npy))/255. 
np.save("../data/final.npy", final_npy)

# Save all information as a pickle file
final_dataset.to_pickle("../data/final_pickle.p")

