import re
import numpy as np


''' define image information as well as its encoding '''

def text_format(font):
    ''' verify the font format (i.e. bold or italics)'''
    txt_format = [0, 0]
    regex_bold = re.findall(r"_B+", font)
    regex_it = re.findall(r"_I+", font)
    if regex_bold:
        txt_format[0] = 1
    if regex_it:
        txt_format[1] = 1
    return txt_format


def clean_font(font):
    ''' clean font names (remove file type and info on bold/italics) '''
    if re.findall(r"(_[A-Z])+.+", font):
        return re.sub(r"(_[A-Z])+.+", "", font)
    else:
        return re.sub(r"\.\w+", "", font)
    
def predict_categorical(lb,dim,predict):
    ''' decode prediction of categorical variable (i.e. font and character) '''
    prediction = np.eye(dim, dtype=int)[np.argmax(predict, axis=1)]
    return lb.inverse_transform(prediction)


def predict_bool(predict):
    ''' decode prediction of boolean variable (i.e. bold and italics) '''
    prediction = predict.reshape(predict.shape[0],)
    return list(map(lambda x: True if x>0.50 else False, prediction))

