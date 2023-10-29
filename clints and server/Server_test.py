import socket
import pickle
import tensorflow as tf
import time

import numpy as np 
import pandas as pd 

import re
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer


# In[5]:


with open('text_input.txt', encoding='utf-8') as file:
    data = file.read()


# In[6]:


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


# In[7]:


ip_dim=len(tokenizer.word_index)+1
ip_dim


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
model=Sequential()
model.add(Embedding(6746,10,input_length=3))
model.add(LSTM(1000,return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(ip_dim,activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.save('server.keras')


'''# In[ ]:


def federated_averaging(global_weights, client_weights_list):
    num_clients = len(client_weights_list)
    averaged_weights = global_weights.copy()
    
    for i in range(len(averaged_weights)):
        # Initialize with the global weights
        averaged_weights[i] = global_weights[i]
        
        # Aggregate model updates from all clients
        for client_weights in client_weights_list:
            averaged_weights[i] += client_weights[i] / num_clients
    
    return averaged_weights


# In[ ]:


# Define the server's IP address and port
server_ip = '192.168.55.103'  # Replace with your server's local IP address
server_port = 12347

# Create a server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the server socket to the IP address and port
server_socket.bind((server_ip, server_port))

# Listen for incoming connections
server_socket.listen(3)

print(f"Server is listening for incoming connections on {server_ip}:{server_port}...")


# Accept connections from multiple clients
# Simulate federated learning rounds
num_rounds = 6
num_clients= 3
    
# Receive model weights from each client
client_weights_list = []

for _ in range(num_clients):
    client_socket, client_address = server_socket.accept()
    print(f"Accepted connection from {client_address}")
    
    client_weights_bytes = b''
    for round_num in range(num_rounds):
        while True:
            data = client_socket.recv(13246848)  # Adjust buffer size as needed
            if not data:
                break
            client_weights_bytes += data
            

        if not client_weights_bytes:
            print(f"No data received from a client{_ + 1}, in round {round_num}.")
            continue

        try:
            client_model_weights = pickle.loads(client_weights_bytes)
            client_weights_list.append(client_model_weights)
            
        except pickle.UnpicklingError as e:
            print(f"Error while unpickling data from a client: {e}")
            
    client_socket.close()
    



# In[ ]:


# Aggregate model updates using federated averaging
if client_weights_list:
    global_weights = federated_averaging(model.get_weights(), client_weights_list)
    model.set_weights(global_weights)
    print(f"Updated global model with data from {num_clients} clients in round {len(client_weights_list)}.")

        
# The global model now contains the federated learning result
model.save_weights('model_weights.h5')

# Close the server socket
server_socket.close()


# In[ ]:


# For preprocessing input text for test
def preprocess(text):
    soup1=BeautifulSoup(text,'html.parser')#remove html tag using beautifulsoup
    non_html_txt=soup.get_text()
    
    #remove unwanted charectors and symbols
    text=re.sub('[^a-zA-Z0-9\s]',' ',non_html_txt)
    text = text.replace('\n', '').replace('\r', '').replace('\ufeff', '')

    #remove extra spaces
    z = []
    for i in text.split():
        if i not in z:
            z.append(i)  
    text = ' '.join(z)
    return text


# In[ ]:


# It gives single predicted words
def predict_nxt_word(txt):

    tok_text = tokenizer.texts_to_sequences([ txt])
    #print(tok_text)
    preds = model.predict(np.array(tok_text), verbose=0)[0]
    next_word = tokenizer.sequences_to_texts([[np.argmax(preds)]])[0]
    return next_word




'''




