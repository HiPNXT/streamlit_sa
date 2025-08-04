import streamlit as st
import torch
import regex
import re
import pandas as pd
import numpy as np
import pickle
from underthesea import sent_tokenize
from pyvi import ViTokenizer, ViPosTagger
from transformers import AutoModel, AutoTokenizer

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

# CHUYỂN ĐỔI EMOJI THÀNH CẢM XÚC TƯƠNG ỨNG, TEENCODE VÀ CÁC TỪ TIẾNG ANH, XÓA CÁC TỪ SAI.
def process_text(text, emoji_dict=emoji_dict, teen_dict=teen_dict, wrong_lst=wrong_lst, english_dict=english_dict):
    document = text.lower()
    document = document.replace("’", '')
    document = regex.sub(r'\.+', ".", document)

    new_sentence = []
    for sentence in sent_tokenize(document):
        # Loại bỏ số và dấu câu, giữ lại từ tiếng Việt
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        words = regex.findall(pattern, sentence)

        cleaned_words = []
        for word in words:
            # Chuyển đổi emoji
            if word in emoji_dict:
                cleaned_words.append(emoji_dict[word])
            # # Chuyển đổi teen code
            elif word in teen_dict:
                cleaned_words.append(teen_dict[word])
            # Xóa từ sai chính tả
            elif word not in wrong_lst:
                cleaned_words.append(word)
            # Chuyển đổi từ tiếng Anh sang tiếng Việt
            elif word in english_dict:
                cleaned_words.append(english_dict[word])

        if cleaned_words:
            new_sentence.append(' '.join(cleaned_words) + '.')

    # Xóa khoảng trắng dư thừa
    document = ' '.join(new_sentence).strip()
    return document

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, đâu...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

def normalize_repeated_characters(text):
    return re.sub(r'(.)\1+', r'\1', text)

def process_postag_pyvi(text):
    new_document = []
    # Các từ loại bạn muốn giữ lại (N: danh từ, A: tính từ, V: động từ, R: trạng từ, v.v.)
    lst_word_type = ['N', 'Np', 'A', 'AB', 'V', 'VB', 'VY', 'R']

    for sentence in sent_tokenize(text):
        # Tách từ (tokenize) bằng Pyvi
        tokenized_sentence = ViTokenizer.tokenize(sentence)

        # POS tagging
        words, tags = ViPosTagger.postagging(tokenized_sentence)

        # Chọn những từ có POS trong danh sách cho phép
        processed_words = [
            word for word, tag in zip(words, tags)
            if tag in lst_word_type
        ]

        new_document.append(" ".join(processed_words))
    return " ".join(new_document)

def remove_stopword(text, stopwords = stopwords_lst):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# Load PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")

def extract_features(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(input_ids)
    mean_vec = outputs.last_hidden_state.mean(dim=1)
    max_vec = outputs.last_hidden_state.max(dim=1).values
    return torch.cat((mean_vec, max_vec), dim=1).squeeze().numpy()  # vector 768*2 = 1536

@st.cache_data
def load_scale():
    with open('model/scaler_phobert.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_data
def load_model():
    # Đọc lại mô hình
    with open('model/svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model

#'Vị trí thuận tiện, sag đg là biển, phòng rộng rãi, sạch sẽ, dịch vụ tốt. Sẽ quay lại'
#'Đặt phòng gđ 2 ngày gồm 3 ng lớn 1 trẻ em mà đến nơi bị phụ thu thêm cho trẻ em. đặt thêm 1 ngày sau đó thì không bị phụ thu'
#'Tự sách vali lên, xuống cầu thang để lên tầng 1 quầy lễ tân, đi với con nhỏ nên rất cực'
content = ''
content = st.text_input(label="Nhập bình luận của bạn:")
if content != '':
    svm_model = load_model()
    scaler = load_scale()
    string_1 = process_text(content)
    string_2 = covert_unicode(string_1)
    string_3 = process_special_word(string_2)
    string_4 = normalize_repeated_characters(string_3)
    string_5 = process_postag_pyvi(string_4)
    string_6 = remove_stopword(string_5)

    X_phobert_test = np.vstack([extract_features(string_6)])
    X_phobert_test = scaler.transform(X_phobert_test)

    y_pred = svm_model.predict(X_phobert_test)
    if(y_pred == 0):
        st.write("Mô hình dự đoán bình luận này là: <TÍCH CỰC>")
    elif(y_pred == 1):
        st.write("Mô hình dự đoán bình luận này là: <TRUNG TÍNH>")
    else:

        st.write("Mô hình dự đoán bình luận này là: <TIÊU CỰC>")
