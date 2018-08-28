import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten,Dropout,ZeroPadding2D
from sklearn.preprocessing import LabelBinarizer
import define_info as di
from contextlib import redirect_stdout

''' load pickle and npy files'''
final_npy = np.load("../data/final.npy")
pickle_info = pd.read_pickle("../data/final_pickle.p")

''' transform font & character into dummy variables (alphabetical order)'''
lb_font=LabelBinarizer()
dummy_font = lb_font.fit_transform(np.array(pickle_info['Font']))
lb_char=LabelBinarizer()
dummy_char = lb_char.fit_transform(np.array(pickle_info['Character']))

'''CNN'''
# Define the shape of the CNN input
input_model = Input((64, 64, 1),name='input_model')
# Perform a Zero-Padding
pad1=ZeroPadding2D(padding=(3,3),data_format="channels_last")(input_model)

# First convolution 5x5
conv1 = Conv2D(32, (5,5), strides=(2,2), data_format="channels_last", activation="relu",padding='same',kernel_initializer='glorot_normal') (pad1)
# First max pooling 2x2
pool1 = MaxPooling2D((2,2), strides=(2,2), data_format="channels_last") (conv1)

# Second convolution 3x3
conv2 = Conv2D(64, (3,3), strides=(1,1), data_format="channels_last", activation="relu",padding='same',kernel_initializer='glorot_normal') (pool1) 
# Second max pooling 2x2
pool2 = MaxPooling2D((2,2), strides=(2,2), data_format="channels_last") (conv2)

# Third convolution 3x3
conv3 = Conv2D(64, (3,3), strides=(1,1), data_format="channels_last", activation="relu",padding='same',kernel_initializer='glorot_normal') (pool2) 
# Third max pooling 2x2
pool3 = MaxPooling2D((2,2), strides=(2,2), data_format="channels_last") (conv3)

# Flatten the output of the third Maxpool layer
input_NN = Flatten() (pool3)
# Randomly drop 50% of the nodes 
dropout1=Dropout(0.5)(input_NN)

'''italics'''
layer1 = Dense(256, activation="relu",kernel_initializer='glorot_normal')(dropout1)
# Randomly drop 40% of the nodes 
dropout2=Dropout(0.4)(layer1)
output_ital = Dense(1, activation="hard_sigmoid",name='output_italics')(layer1) 

'''bold'''
layer2 = Dense(256, activation="relu",kernel_initializer='glorot_normal')(dropout2)
# Randomly drop 40% of the nodes 
dropout3=Dropout(0.4)(layer2)
output_bold = Dense(1, activation="hard_sigmoid",name='output_bold')(layer2) 

'''char'''
layer3 = Dense(128, activation="relu",kernel_initializer='glorot_normal')(dropout3)
# Randomly drop 40% of the nodes 
dropout4=Dropout(0.4)(layer3)
output_char = Dense(94, activation="softmax",name='output_char')(layer3) 

''' font '''
layer4 = Dense(64, activation="relu",kernel_initializer='glorot_normal')(dropout4)
 # Dense(11) since they are eleven distinct fonts (excluding _B and _I)
output_font = Dense(11, activation="softmax",name='output_font')(layer4)

''' Model'''
model = Model(inputs=input_model, outputs=[output_font,output_char, output_bold, output_ital])

model.compile( loss={'output_font': 'categorical_crossentropy', 'output_char': 'categorical_crossentropy','output_bold': 'binary_crossentropy', 'output_italics': 'binary_crossentropy'},
              optimizer = 'adadelta',metrics = ['accuracy'], loss_weights={'output_font': 0.3,'output_char':0.3, 'output_bold':0.2,'output_italics':0.2})

model.fit({'input_model': final_npy},
          {'output_font': dummy_font, 'output_char': dummy_char, 'output_bold': pickle_info['Bold'], 'output_italics': pickle_info['Italics']}, epochs = 40, batch_size=64)

'''Predictions'''
prediction=model.predict(final_npy)
pred_font=di.predict_categorical(lb_font,11,prediction[0])
pred_char=di.predict_categorical(lb_char,94,prediction[1])
pred_bold=di.predict_bool(prediction[2])
pred_ital=di.predict_bool(prediction[3])

'''Save model and predictions'''
prediction_csv=pd.DataFrame({'Character':pred_char,'Font':pred_font,'Bold':pred_bold,'Italics':pred_ital})
prediction_csv.to_csv("predictions.csv",header=None)

'''Save summary of the model'''
with open('../data_out/model/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()


'''Save modelto json and h5'''
model_json = model.to_json()
with open("../data_out/model/model.json","w") as json_file:
    json_file.write(model_json)
    
model.save_weights("../data_out/model/model.h5")




