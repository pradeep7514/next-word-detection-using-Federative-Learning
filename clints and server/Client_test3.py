#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np 
import pandas as pd 

import re
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer


# In[24]:


with open('text_input.txt', encoding='utf-8') as file:
    data = file.read()


# In[25]:


soup=BeautifulSoup(data,'html.parser')#remove html tag using beautifulsoup
non_html_text=soup.get_text()
    
#remove unwanted charectors and symbols
text=re.sub('[^a-zA-Z0-9\s]',' ',non_html_text)
text = text.replace('\n', '').replace('\r', '').replace('\ufeff', '')

#remove extra spaces
z = []
for i in text.split():
    if i not in z:
        z.append(i)  
text = ' '.join(z)

#tokenize text
    
tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])
sequence=tokenizer.texts_to_sequences([text])[0]

sequence[:15]


# In[26]:


ip_dim=len(tokenizer.word_index)+1
ip_dim


# In[27]:


#traning sequence

sequences = []

for i in range(3, len(sequence)):
    words = sequence[i-3:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)


x,y=[],[]
for i in sequences:
    x.append(i[0:3])
    y.append(i[3])
x=np.array(x)
y=np.array(y)


# In[28]:


x[:10],y[:10]


# In[29]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# # # Pad "sequences" to a fixed length
# max_len = 10
# x = pad_sequences(x, maxlen=max_len)
# y = pad_sequences(y, maxlen=max_len)

# Convert y to categorical

y = to_categorical(y, num_classes=ip_dim)
y[:10]


# In[30]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
model=Sequential()
model.add(Embedding(ip_dim,10,input_length=3))
model.add(LSTM(1000,return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(ip_dim,activation='softmax'))
model.summary()


# In[31]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[32]:


model.fit(x,y,batch_size=62,epochs=1)


# In[39]:
model.save("client3.keras")

'''import socket
import pickle
import tensorflow as tf
from sharedconfig import vocab_size, embedding_dim
import numpy as np

# Define the server's IP address and port
server_ip = '192.168.94.137'  # Replace with your server's local IP address
server_port = 12347

# Create a client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((server_ip, server_port))


# In[40]:


# Get the client's model weights
client_model_weights = model.get_weights()

#save the weights
model.save_weights('l_model3_weights.h5')

# Serialize and send the model weights to the server
client_weights_bytes = pickle.dumps(client_model_weights)
client_socket.send(client_weights_bytes)


# In[41]:


#print(client_model_weights)


# In[42]:


# Close the client socket
client_socket.close()


# In[ ]:





# In[ ]:





# In[ ]:'''




