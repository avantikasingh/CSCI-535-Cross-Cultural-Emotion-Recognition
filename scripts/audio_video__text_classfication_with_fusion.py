#!/usr/bin/env python
# coding: utf-8

# In[1]:


#model for audio to text


# In[2]:


import os
import torch
import pandas as pd

def extract_label_from_filename(filename):
    # Extract the second and third characters from the filename
    label = filename[1:3]
    return label

def read_tensors_from_folder(folder_path):
    df = pd.DataFrame(columns=['Category','Tensor_File', 'Label'])
    labels = []
    filenames = []
    category = []
    
    
    # Iterate through each file
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if '_C1_' in file:
            category = 'C1'
        elif '_C2_' in file:
            category = 'C2'
        elif '_C3_' in file:
            category = 'C3'
        elif '_C4_' in file:
            category = 'C4'
        elif '_C5_' in file:
            category = 'C5'
        elif '_C6_' in file:
            category = 'C6'
        else:
            category = 'unknown'
          
        # Extract label from the file name
        label = extract_label_from_filename(file)
        # Append information to DataFrame
        df.loc[len(df)] = [category,file_path,label]
        
    return df


# Example usage:
folder_path = "./HuBERT_extracted_features"
df = read_tensors_from_folder(folder_path)
print(df.head())


# In[3]:


import torch

# Specify the path to the .pt file
file_path = './HuBERT_extracted_features/SAH_C1_S018_P036_VC1_001394_002328.pt'  # Replace with the actual path to your .pt file

# Load the .pt file
embeddings = torch.load(file_path)

# Now, 'embeddings' variable contains the loaded PyTorch tensor
print(embeddings)
print(embeddings.shape)


# In[4]:


df_C1 = df[df['Category']=='C1']
df_C2 = df[df['Category']=='C2']
df_C3 = df[df['Category']=='C3']
df_C4 = df[df['Category']=='C4']
df_C5 = df[df['Category']=='C5']
df_C6 = df[df['Category']=='C6']


# In[5]:


df_C1_Arousal = df_C1[df_C1['Label'].str.contains('AH|AL', case=False)]
print(len(df_C1_Arousal))
df_C1_Valence = df_C1[df_C1['Label'].str.contains('VH|VL', case=False)]
df_C1_Liking  =  df_C1[df_C1['Label'].str.contains('SD|SL', case=False)]

df_C2_Arousal = df_C2[df_C2['Label'].str.contains('AH|AL', case=False)]
df_C2_Valence = df_C2[df_C2['Label'].str.contains('VH|VL', case=False)]
df_C2_Liking  = df_C2[df_C2['Label'].str.contains('SD|SL', case=False)]

df_C3_Arousal = df_C3[df_C3['Label'].str.contains('AH|AL', case=False)]
df_C3_Valence = df_C3[df_C3['Label'].str.contains('VH|VL', case=False)]
df_C3_Liking  = df_C3[df_C3['Label'].str.contains('SD|SL', case=False)]

df_C4_Arousal = df_C4[df_C4['Label'].str.contains('AH|AL', case=False)]
df_C4_Valence = df_C4[df_C4['Label'].str.contains('VH|VL', case=False)]
df_C4_Liking  = df_C4[df_C4['Label'].str.contains('SD|SL', case=False)]

df_C5_Arousal = df_C5[df_C5['Label'].str.contains('AH|AL', case=False)]
df_C5_Valence = df_C5[df_C5['Label'].str.contains('VH|VL', case=False)]
df_C5_Liking  = df_C5[df_C5['Label'].str.contains('SD|SL', case=False)]

df_C6_Arousal = df_C6[df_C6['Label'].str.contains('AH|AL', case=False)]
df_C6_Valence = df_C6[df_C6['Label'].str.contains('VH|VL', case=False)]
df_C6_Liking  = df_C6[df_C6['Label'].str.contains('SD|SL', case=False)]


# In[6]:


print(len(df_C6_Liking))


# In[7]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import torch

# Load your DataFrame
data = df_C5

# Load and preprocess the BERT embeddings
X = [torch.load(file_path) for file_path in data['Tensor_File']]

# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print("Number of samples in X_train:", len(X_train))
print("Number of samples in y_train:", len(y_train))
print("Number of samples in X_val:", len(X_val))
print("Number of samples in y_val:", len(y_val))
print("Number of samples in X_test:", len(X_test))
print("Number of samples in y_test:", len(y_test))


# In[8]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import torch

def train_simple_rnn_model(df_train, df_test):
    # Load and preprocess the BERT embeddings
    X_en_train = [torch.load(file_path) for file_path in df_train['Tensor_File']]
    X_en_test = [torch.load(file_path) for file_path in df_test['Tensor_File']]


    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['Label'])
    y_test = label_encoder.fit_transform(df_test['Label'])

    # Determine the maximum sequence length
    max_length_train = max(len(embedding) for embedding in X_en_train)
    max_length_test = max(len(embedding) for embedding in X_en_test)


    # Pad or truncate the embeddings to the maximum length
    X_train = [np.pad(embedding, ((0, max_length_train - len(embedding)), (0, 0)), mode='constant') for embedding in X_en_train]
    X_test = [np.pad(embedding, ((0, max_length_test - len(embedding)), (0, 0)), mode='constant') for embedding in X_en_test]

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Define the Simple RNN model
    model = Sequential([
        SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2])),  # 64 is the number of units in the RNN layer
        Dense(64, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax activation for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    print(model.summary())

    # Train the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    
    new_data_predictions = model.predict(X_test)

    # Inverse transform the predictions to get the original labels
    predicted_labels = label_encoder.inverse_transform(np.argmax(new_data_predictions, axis=1))

    # Print or use the predicted labels as needed
    print(predicted_labels)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    return  predicted_labels, y_test



# In[9]:


combined_df_Arousal = pd.concat([df_C1_Arousal, df_C4_Arousal], ignore_index=True)
combined_df_Valence = pd.concat([df_C1_Arousal, df_C4_Arousal], ignore_index=True)
combined_df_Liking = pd.concat([df_C1_Arousal, df_C4_Arousal], ignore_index=True)


# In[10]:


print("Training for C2 audio")

audio_c2_arousal_lables, audio_c2_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C2_Arousal)
audio_c2_valence_lables, audio_c2_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C2_Valence)
audio_c2_liking_lables, audio_c2_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C2_Liking)


# In[10]:


print("Training for C3 audio")

audio_c3_arousal_lables, audio_c3_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C3_Arousal)
audio_c3_valence_lables, audio_c3_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C3_Valence)
audio_c3_liking_lables, audio_c3_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C3_Liking)



# In[11]:


print("Training for C5 audio")

audio_c5_arousal_lables, audio_c5_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C5_Arousal)
audio_c5_valence_lables, audio_c5_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C5_Valence)
audio_c5_liking_lables, audio_c5_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C5_Liking)



# In[12]:


print("Training for C6 audio")

audio_c6_arousal_lables, audio_c6_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C6_Arousal)
audio_c6_valence_lables, audio_c6_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C6_Valence)
audio_c6_liking_lables, audio_c6_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C6_Liking)



# In[15]:


#video features classification


# In[6]:


import os
import torch
import pandas as pd

def extract_label_from_filename(filename):
    # Extract the second and third characters from the filename
    label = filename[1:3]
    return label

def read_tensors_from_folder(folder_path):
    df = pd.DataFrame(columns=['Category','Tensor_File', 'Label'])
    labels = []
    filenames = []
    category = []
    
    
    # Iterate through each file
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if '_C1_' in file:
            category = 'C1'
        elif '_C2_' in file:
            category = 'C2'
        elif '_C3_' in file:
            category = 'C3'
        elif '_C4_' in file:
            category = 'C4'
        elif '_C5_' in file:
            category = 'C5'
        elif '_C6_' in file:
            category = 'C6'
        else:
            category = 'unknown'
          
        # Extract label from the file name
        label = extract_label_from_filename(file)
        # Append information to DataFrame
        df.loc[len(df)] = [category,file_path,label]
        
    return df


# Example usage:
folder_path = "./vggpooled"
df = read_tensors_from_folder(folder_path)
print(df.head())


# In[7]:


import torch

# Specify the path to the .pt file
file_path = './video_features/SVH_C6_S169_P338_VC1_000002_000855.pt'  # Replace with the actual path to your .pt file

# Load the .pt file
embeddings = torch.load(file_path)

# Now, 'embeddings' variable contains the loaded PyTorch tensor
print(embeddings)
print(embeddings.shape)


# In[8]:


df_C1_video = df[df['Category']=='C1']
df_C2_video = df[df['Category']=='C2']
df_C3_video = df[df['Category']=='C3']
df_C4_video = df[df['Category']=='C4']
df_C5_video = df[df['Category']=='C5']
df_C6_video = df[df['Category']=='C6']


# In[9]:


df_C1_Arousal = df_C1_video[df_C1_video['Label'].str.contains('AH|AL', case=False)]
print(len(df_C1_Arousal))
df_C1_Valence = df_C1_video[df_C1_video['Label'].str.contains('VH|VL', case=False)]
df_C1_Liking  =  df_C1_video[df_C1_video['Label'].str.contains('SD|SL', case=False)]

df_C2_Arousal = df_C2_video[df_C2_video['Label'].str.contains('AH|AL', case=False)]
df_C2_Valence = df_C2_video[df_C2_video['Label'].str.contains('VH|VL', case=False)]
df_C2_Liking  = df_C2_video[df_C2_video['Label'].str.contains('SD|SL', case=False)]

df_C3_Arousal = df_C3_video[df_C3_video['Label'].str.contains('AH|AL', case=False)]
df_C3_Valence = df_C3_video[df_C3_video['Label'].str.contains('VH|VL', case=False)]
df_C3_Liking  = df_C3_video[df_C3_video['Label'].str.contains('SD|SL', case=False)]

df_C4_Arousal = df_C4_video[df_C4_video['Label'].str.contains('AH|AL', case=False)]
df_C4_Valence = df_C4_video[df_C4_video['Label'].str.contains('VH|VL', case=False)]
df_C4_Liking  = df_C4_video[df_C4_video['Label'].str.contains('SD|SL', case=False)]

df_C5_Arousal = df_C5_video[df_C5_video['Label'].str.contains('AH|AL', case=False)]
df_C5_Valence = df_C5_video[df_C5_video['Label'].str.contains('VH|VL', case=False)]
df_C5_Liking  = df_C5_video[df_C5_video['Label'].str.contains('SD|SL', case=False)]

df_C6_Arousal = df_C6_video[df_C6_video['Label'].str.contains('AH|AL', case=False)]
df_C6_Valence = df_C6_video[df_C6_video['Label'].str.contains('VH|VL', case=False)]
df_C6_Liking  = df_C6_video[df_C6_video['Label'].str.contains('SD|SL', case=False)]


# In[ ]:


combined_df_Arousal = pd.concat([df_C1_Arousal, df_C4_Arousal], ignore_index=True)
combined_df_Valence = pd.concat([df_C1_Arousal, df_C4_Arousal], ignore_index=True)
combined_df_Liking = pd.concat([df_C1_Arousal, df_C4_Arousal], ignore_index=True)


# In[20]:


print("Training for C2 audio")

video_c2_arousal_lables, video_c2_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C2_Arousal)
video_c2_valence_lables, video_c2_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C2_Valence)
video_c2_liking_lables, video_c2_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C2_Liking)



# In[21]:


print("Training for C3 audio")

video_c3_arousal_lables, video_c3_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C3_Arousal)
video_c3_valence_lables, video_c3_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C3_Valence)
video_c3_liking_lables, video_c3_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C3_Liking)



# In[22]:


print("Training for C5 audio")

video_c5_arousal_lables, video_c5_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C5_Arousal)
video_c5_valence_lables, video_c5_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C5_Valence)
video_c5_liking_lables, video_c5_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C5_Liking)



# In[23]:


print("Training for C6 audio")

video_c6_arousal_lables, video_c6_arousal_ground_truth = train_simple_rnn_model(combined_df_Arousal,df_C6_Arousal)
video_c6_valence_lables, video_c6_valence_ground_truth = train_simple_rnn_model(combined_df_Valence,df_C6_Valence)
video_c6_liking_lables, video_c6_liking_ground_truth = train_simple_rnn_model(combined_df_Liking,df_C6_Liking)


# In[26]:


# video classification marlin 


# In[10]:


# import os
# import torch
# import pandas as pd

# def extract_label_from_filename(filename):
#     # Extract the second and third characters from the filename
#     label = filename[1:3]
#     return label

# def read_tensors_from_folder(folder_path):
#     df = pd.DataFrame(columns=['Category','Tensor_File', 'Label'])
#     labels = []
#     filenames = []
#     category = []
    
    
#     # Iterate through each file
#     for file in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file)
#         if '_C1_' in file:
#             category = 'C1'
#         elif '_C2_' in file:
#             category = 'C2'
#         elif '_C3_' in file:
#             category = 'C3'
#         elif '_C4_' in file:
#             category = 'C4'
#         elif '_C5_' in file:
#             category = 'C5'
#         elif '_C6_' in file:
#             category = 'C6'
#         else:
#             category = 'unknown'
          
#         # Extract label from the file name
#         label = extract_label_from_filename(file)
#         # Append information to DataFrame
#         df.loc[len(df)] = [category,file_path,label]
        
#     return df


# # Example usage:
# folder_path = "./video_features"
# df = read_tensors_from_folder(folder_path)
# print(df.head())


# In[11]:


# df_C1_video_marlin = df[df['Category']=='C1']
# df_C2_video_marlin = df[df['Category']=='C2']
# df_C3_video_marlin = df[df['Category']=='C3']
# df_C4_video_marlin = df[df['Category']=='C4']
# df_C5_video_marlin = df[df['Category']=='C5']
# df_C6_video_marlin = df[df['Category']=='C6']


# In[12]:


# df_C1_Arousal =  df_C1_video_marlin[df_C1_video_marlin['Label'].str.contains('AH|AL', case=False)]
# print(len(df_C1_Arousal))
# df_C1_Valence =  df_C1_video_marlin[df_C1_video_marlin['Label'].str.contains('VH|VL', case=False)]
# df_C1_Liking  =  df_C1_video_marlin[df_C1_video_marlin['Label'].str.contains('SD|SL', case=False)]

# df_C2_Arousal = df_C2_video_marlin[df_C2_video_marlin['Label'].str.contains('AH|AL', case=False)]
# df_C2_Valence = df_C2_video_marlin[df_C2_video_marlin['Label'].str.contains('VH|VL', case=False)]
# df_C2_Liking  = df_C2_video_marlin[df_C2_video_marlin['Label'].str.contains('SD|SL', case=False)]

# df_C3_Arousal = df_C3_video_marlin[df_C3_video_marlin['Label'].str.contains('AH|AL', case=False)]
# df_C3_Valence = df_C3_video_marlin[df_C3_video_marlin['Label'].str.contains('VH|VL', case=False)]
# df_C3_Liking  = df_C3_video_marlin[df_C3_video_marlin['Label'].str.contains('SD|SL', case=False)]

# df_C4_Arousal = df_C4_video_marlin[df_C4_video_marlin['Label'].str.contains('AH|AL', case=False)]
# df_C4_Valence = df_C4_video_marlin[df_C4_video_marlin['Label'].str.contains('VH|VL', case=False)]
# df_C4_Liking  = df_C4_video_marlin[df_C4_video_marlin['Label'].str.contains('SD|SL', case=False)]

# df_C5_Arousal = df_C5_video_marlin[df_C5_video_marlin['Label'].str.contains('AH|AL', case=False)]
# df_C5_Valence = df_C5_video_marlin[df_C5_video_marlin['Label'].str.contains('VH|VL', case=False)]
# df_C5_Liking  = df_C5_video_marlin[df_C5_video_marlin['Label'].str.contains('SD|SL', case=False)]

# df_C6_Arousal = df_C6_video_marlin[df_C6_video_marlin['Label'].str.contains('AH|AL', case=False)]
# df_C6_Valence = df_C6_video_marlin[df_C6_video_marlin['Label'].str.contains('VH|VL', case=False)]
# df_C6_Liking  = df_C6_video_marlin[df_C6_video_marlin['Label'].str.contains('SD|SL', case=False)]


# In[30]:


# print("Training for C1 video")
# train_simple_rnn_model(df_C1_Arousal)
# train_simple_rnn_model(df_C1_Valence)
# train_simple_rnn_model(df_C1_Liking)


# In[31]:


# print("Training for C2 video")
# train_simple_rnn_model(df_C2_Arousal)
# train_simple_rnn_model(df_C2_Valence)
# train_simple_rnn_model(df_C2_Liking)


# In[32]:


# print("Training for C3 video")
# train_simple_rnn_model(df_C3_Arousal)
# train_simple_rnn_model(df_C3_Valence)
# train_simple_rnn_model(df_C3_Liking)


# In[33]:


# print("Training for C4 video")
# train_simple_rnn_model(df_C4_Arousal)
# train_simple_rnn_model(df_C4_Valence)
# train_simple_rnn_model(df_C4_Liking)


# In[34]:


# print("Training for C5 video")
# train_simple_rnn_model(df_C5_Arousal)
# train_simple_rnn_model(df_C5_Valence)
# train_simple_rnn_model(df_C5_Liking)


# In[35]:


# print("Training for C6 video")
# train_simple_rnn_model(df_C6_Arousal)
# train_simple_rnn_model(df_C6_Valence)
# train_simple_rnn_model(df_C6_Liking)


# In[ ]:


# text modelling


# In[13]:


# 



# In[14]:


# import pandas as pd

# file_path = 'final_translated.csv' 
# df = pd.read_csv(file_path, sep=';')
# df = df[['Category', 'Folder', 'Translated_Text']]
# # Extract last two letters of the first three letters from 'Folder'
# df['Label'] = df['Folder'].str[:3].apply(lambda x: x[-2:].upper())
# print(df.head())


# In[15]:


# from sklearn.model_selection import train_test_split
# df_C1 = df[df['Category'] == 'C1']
# df_C2 = df[df['Category'] == 'C2']
# df_C3 = df[df['Category'] == 'C3']
# df_C4 = df[df['Category'] == 'C4']
# df_C5 = df[df['Category'] == 'C5']
# df_C6 = df[df['Category'] == 'C6']


# In[16]:


# df_C1_Arousal = df_C1[df_C1['Label'].str.contains('AH|AL', case=False)]
# df_C1_Valence = df_C1[df_C1['Label'].str.contains('VH|VL', case=False)]
# df_C1_Liking  =  df_C1[df_C1['Label'].str.contains('SD|SL', case=False)]

# df_C2_Arousal = df_C2[df_C2['Label'].str.contains('AH|AL', case=False)]
# df_C2_Valence = df_C2[df_C2['Label'].str.contains('VH|VL', case=False)]
# df_C2_Liking  = df_C2[df_C2['Label'].str.contains('SD|SL', case=False)]

# df_C3_Arousal = df_C3[df_C3['Label'].str.contains('AH|AL', case=False)]
# df_C3_Valence = df_C3[df_C3['Label'].str.contains('VH|VL', case=False)]
# df_C3_Liking  = df_C3[df_C3['Label'].str.contains('SD|SL', case=False)]

# df_C4_Arousal = df_C4[df_C4['Label'].str.contains('AH|AL', case=False)]
# df_C4_Valence = df_C4[df_C4['Label'].str.contains('VH|VL', case=False)]
# df_C4_Liking  = df_C4[df_C4['Label'].str.contains('SD|SL', case=False)]

# df_C5_Arousal = df_C5[df_C5['Label'].str.contains('AH|AL', case=False)]
# df_C5_Valence = df_C5[df_C5['Label'].str.contains('VH|VL', case=False)]
# df_C5_Liking  = df_C5[df_C5['Label'].str.contains('SD|SL', case=False)]

# df_C6_Arousal = df_C6[df_C6['Label'].str.contains('AH|AL', case=False)]
# df_C6_Valence = df_C6[df_C6['Label'].str.contains('VH|VL', case=False)]
# df_C6_Liking  = df_C6[df_C6['Label'].str.contains('SD|SL', case=False)]


# In[35]:


# def preprocess_data(df):
#     X = df['Translated_Text']
#     y = df['Label']
#     # Split the data into training and validation sets (80% train, 20% validation)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print("Number of samples in X_train:", len(X_train))
#     print("Number of samples in y_train:", len(y_train))
#     print("Number of samples in X_test:", len(X_test))
#     print("Number of samples in y_test:", len(y_test))
#     tokenized_train = tokenizer(X_train.tolist(), padding = True, truncation = True, return_tensors="pt")
#     tokenized_test = tokenizer(X_test.tolist() , padding = True, truncation = True,  return_tensors="pt")
    
#     #move on device (GPU)
#     tokenized_train = {k:torch.tensor(v).to(device) for k,v in tokenized_train.items()}
#     tokenized_test = {k:torch.tensor(v).to(device) for k,v in tokenized_test.items()}
#     return tokenized_train, tokenized_test, X_train, X_test, y_train, y_test


# In[48]:


# import torch
# import torch.nn as nn
# from transformers import BertModel
# from sklearn.preprocessing import LabelEncoder

# def train_bert_classifier(X_train, y_train, tokenized_train, tokenized_test, num_epochs=2):    # Fit the label encoder on the training labels and transform them
#     label_encoder = LabelEncoder()
#     learning_rate = 1e-5
#     num_classes = 2
#     train_labels = label_encoder.fit_transform(y_train)
#     train_labels_tensor = torch.tensor(train_labels).to(device)

#     # Define your classification model
#     class BertClassifier(nn.Module):
#         def __init__(self, num_classes):
#             super(BertClassifier, self).__init__()
#             self.bert = BertModel.from_pretrained('bert-base-cased')
#             self.dropout = nn.Dropout(0.1)
#             self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

#         def forward(self, input_ids, attention_mask):
#             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#             pooled_output = outputs.pooler_output
#             pooled_output = self.dropout(pooled_output)
#             logits = self.fc(pooled_output)
#             return logits

#     # Initialize your model, optimizer, and loss function
#     model = BertClassifier(num_classes=len(label_encoder.classes_)).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     # Training loop
#     for epoch in range(num_epochs):
#         model.train()
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#         optimizer.zero_grad()
#         outputs = model(tokenized_train['input_ids'], tokenized_train['attention_mask'])
#         loss = criterion(outputs, train_labels_tensor)
#         loss.backward()
#         optimizer.step()

#         # Print training loss
#         print(f'Training Loss: {loss.item():.4f}')

#     # Evaluate the model on the test set
#     model.eval()
#     with torch.no_grad():
#         test_outputs = model(tokenized_test['input_ids'], tokenized_test['attention_mask'])
#         test_probabilities = torch.nn.functional.softmax(test_outputs, dim=1)
#         test_predicted_labels = torch.argmax(test_probabilities, dim=1)
#         test_accuracy = torch.sum(test_predicted_labels == test_labels_tensor).item() / len(test_labels_tensor)

#     print(f'Test Accuracy: {test_accuracy:.4f}')

#     # Return the trained model
#     return test_predicted_labels 

# # Example usage:
# # Assuming you have tokenized train and test data in the same format as tokenized_train and tokenized_test
# # Also, assuming X_train, y_train, X_test, y_test are available

# # Train the BERT classifier
# # trained_model = train_bert_classifier(X_train, y_train, tokenized_train, tokenized_test)


# In[49]:


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Assuming you have obtained the predictions and ground truth labels
# # Convert predictions and labels to numpy arrays
# def cal_scores(predicted_labels,train_labels_tensor):
#     predicted_labels_np = predicted_labels.cpu().numpy()
#     true_labels_np = train_labels_tensor.cpu().numpy()

# # Calculate metrics
#     accuracy = accuracy_score(true_labels_np, predicted_labels_np)
#     precision = precision_score(true_labels_np, predicted_labels_np, average='macro')
#     recall = recall_score(true_labels_np, predicted_labels_np, average='macro')
#     f1 = f1_score(true_labels_np, predicted_labels_np, average='macro')

#     print(f'Accuracy: {accuracy:.4f}')
#     print(f'Precision: {precision:.4f}')
#     print(f'Recall: {recall:.4f}')
#     print(f'F1 Score: {f1:.4f}')

# # Compute confusion matrix
#     cm = confusion_matrix(true_labels_np, predicted_labels_np)

# # Plot confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()


# In[50]:


# print("training for C1" )
# tokenized_train, tokenized_val, X_train, X_val, y_train, y_val = preprocess_data(df_C1_Arousal)
# test_predicted_labels = train_bert_classifier(X_train, y_train, tokenized_train, 2)
# print(test_predicted_labels)
# print(y_val)
# tokenized_train, tokenized_val, X_train, X_val, y_train, y_val = preprocess_data(df_C1_Valence)
# train_bert_classifier(X_train, y_train, tokenized_train, 10)
# tokenized_train, tokenized_val, X_train, X_val, y_train, y_val = preprocess_data(df_C1_Liking)
# train_bert_classifier(X_train, y_train, tokenized_train, 10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def combine_predictions(predictions_audio, predictions_video):
    label_encoder = LabelEncoder()

    # Fit label encoder on combined labels (audio + video)
    combined_labels = np.concatenate([predictions_audio, predictions_video])
    label_encoder.fit(combined_labels)

    # Transform string labels to numerical values
    pred_audio_encoded = label_encoder.transform(predictions_audio)
    pred_video_encoded = label_encoder.transform(predictions_video)
    
    combined_predictions = []
    for pred_audio, pred_video in zip(pred_audio_encoded, pred_video_encoded):
        # Combine predictions for each sample using majority vote
        combined_pred = np.argmax(np.bincount([pred_audio, pred_video]))
        combined_predictions.append(combined_pred)
    return combined_predictions


# In[ ]:


for culture in range(1, 7):
    if culture in [2, 3, 6, 5]:
        audio_labels = locals()[f"audio_c{culture}_arousal_lables"]
        video_labels = locals()[f"video_c{culture}_arousal_lables"]
        audio_ground_truth = locals()[f"audio_c{culture}_arousal_ground_truth"]
        video_ground_truth = locals()[f"video_c{culture}_arousal_ground_truth"]

        combined_predictions = combine_predictions(audio_labels, video_labels)

        # Calculate accuracy
        ground_truth_combined = []
        ground_truth_combined.extend(audio_ground_truth)
        ground_truth_combined.extend(video_ground_truth)
        ground_truth_combined = ground_truth_combined[:len(combined_predictions)]
        accuracy = accuracy_score(ground_truth_combined, combined_predictions)
        print(f"Combined accuracy for C{culture} arousal: {accuracy}")


# In[ ]:


for culture in range(1, 7):
    if culture in [2, 3, 6, 5]:
    

        audio_labels = locals()[f"audio_c{culture}_valence_lables"]
        video_labels = locals()[f"video_c{culture}_valence_lables"]
        audio_ground_truth = locals()[f"audio_c{culture}_valence_ground_truth"]
        video_ground_truth = locals()[f"video_c{culture}_valence_ground_truth"]

        combined_predictions = combine_predictions(audio_labels, video_labels)

        # Calculate accuracy
        ground_truth_combined = []
        ground_truth_combined.extend(audio_ground_truth)
        ground_truth_combined.extend(video_ground_truth)
        ground_truth_combined = ground_truth_combined[:len(combined_predictions)]
        accuracy = accuracy_score(ground_truth_combined, combined_predictions)
        print(f"Combined accuracy for C{culture} valence: {accuracy}")


# In[ ]:


for culture in range(1, 7):
    if culture in [2, 3, 6, 5]:
        audio_labels = locals()[f"audio_c{culture}_liking_lables"]
        video_labels = locals()[f"video_c{culture}_liking_lables"]
        audio_ground_truth = locals()[f"audio_c{culture}_liking_ground_truth"]
        video_ground_truth = locals()[f"video_c{culture}_liking_ground_truth"]

        combined_predictions = combine_predictions(audio_labels, video_labels)

        # Calculate accuracy
        ground_truth_combined = []
        ground_truth_combined.extend(audio_ground_truth)
        ground_truth_combined.extend(video_ground_truth)
        ground_truth_combined = ground_truth_combined[:len(combined_predictions)]
        accuracy = accuracy_score(ground_truth_combined, combined_predictions)
        print(f"Combined accuracy for C{culture} liking: {accuracy}")

