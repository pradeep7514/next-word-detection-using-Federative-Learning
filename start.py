###############################**********************************************************#############################
#LOCAL SERVERS TO GLOBAL SERVER
from flask import Flask, request, render_template
import socket
import pickle
import threading
from keras.models import load_model

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

from twilio.rest import Client
import random

from sharedconfig import userver,pserver,username1,password1,username2,password2,username3,password3


validation1=False
validation2=False
validation3=False
validation4=False


app = Flask('__name__')
app.secret_key = 'otp'

# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model1 = load_model('client1.keras', custom_objects=custom_objects)

# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model2 = load_model('client2.keras', custom_objects=custom_objects)

# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model3 = load_model('client3.keras', custom_objects=custom_objects)


# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model = load_model('server.keras', custom_objects=custom_objects)

# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

global_model = load_model('global_model.keras', custom_objects=custom_objects)


@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/authenticate', methods=['GET', 'POST'])
def authentication_m():
    return render_template('Mobile_auth.html')

@app.route('/mobile_auth', methods = ['GET', 'POST'])
def getOTP():
    number = request.form['number']
    getOTPApi(number)
    val = getOTPApi(number)
    if val:
        return render_template('authentication.html')

def generateOTP():
    return random.randrange(1000, 9999)

def getOTPApi(number):
    global session
    session = dict()
    account_sid = 'AC7c21ea6c452237b0cdfa2b542e12beda'
    auth_token = '664b89a7fe8b6b4a87240391fbb45e9b'
    client = Client(account_sid, auth_token)
    otp = generateOTP()
    session['response'] = str(otp)
    body = 'Your OTP is '+str(otp)
    message = client.messages.create(
        from_ = "+16562230550",
        body = body,
        to = number
    )
    if message.sid:
        return True
    else:
        return False


@app.route('/authenticate1', methods=['GET', 'POST'])
def authentication_p1():

    uname=request.form["username"]
    passw=request.form["password"]
    cid=request.form["CID"]
    otp = request.form['otp']
    
    if(cid=='0'):
        if(uname==userver and passw==pserver): 
            if 'response' in session:
                s = session['response']
                session.pop('response', None)
                if s == otp:
                    global validation1
                    validation1=True
                    return render_template('welcome.html')
                else:
                    msg="WRONG OTP"
                    return render_template('error.html',n=msg)      
        else:
            msg="NOT A VALID USER"
            return render_template('error.html',n=msg)
            

    elif(cid=='1'):
        if(uname==username1 and passw==password1):
            if 'response' in session:
                s = session['response']
                session.pop('response', None)
                if s == otp:
                    global validation2
                    validation2=True
                    return render_template('welcome.html')
                else:
                    msg="WRONG OTP"
                    return render_template('error.html',n=msg)      
        else:
            msg="NOT A VALID USER"
            return render_template('error.html',n=msg)

    elif(cid=='2'):
        if(uname==username2 and passw==password2):
            if 'response' in session:
                s = session['response']
                session.pop('response', None)
                if s == otp:
                    global validation3
                    validation3s=True
                    return render_template('welcome.html')
                else:
                    msg="WRONG OTP"
                    return render_template('error.html',n=msg)      
        else:
            msg="NOT A VALID USER"
            return render_template('error.html',n=msg)

    elif(cid=='3'):
        if(uname==username3 and passw==password3):
            if 'response' in session:
                s = session['response']
                session.pop('response', None)
                if s == otp:
                    global validation4
                    validation4=True
                    return render_template('welcome.html')
                else:
                    msg="WRONG OTP"
                    return render_template('error.html',n=msg)      
        else:
            msg="NOT A VALID USER"
            return render_template('error.html',n=msg)
            

    #return "Not valid"    


'''#compilation phase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
model=Sequential()
model.add(Embedding(6746,10,input_length=3))
model.add(LSTM(1000,return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(6746,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])'''

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

@app.route('/server', methods=['GET', 'POST'])
def server_active():
    if(validation1==True):

        # Define the server's IP address and port
        server_ip = '10.10.16.114'  # Replace with your server's local IP address
        server_port = 12347

        # Create a server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the server socket to the IP address and port
        server_socket.bind((server_ip, server_port))

        # Listen for incoming connections
        server_socket.listen(3)


        # Accept connections from multiple clients
        num_rounds = 6
        num_clients = 3
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
            

        # Aggregate model updates using federated averaging
        if client_weights_list:
            global_weights = federated_averaging(model.get_weights(), client_weights_list)
            model.set_weights(global_weights)
            print(f"Updated global model with data from {num_clients} clients in round {len(client_weights_list)}.")
        
        # The global model now contains the federated learning result
        model.save('global_model.keras')

        
        # Close the server socket
        server_socket.close()

        return render_template("welcome.html")

    else:
        msg="Not Valid User"
        return render_template('error.html',n=msg)
    

def client_worker(model, client_num):
    server_ip = '10.10.16.114'  # Replace with your server's local IP address
    server_port = 12347
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    client_model_weights = model.get_weights()
    client_weights_bytes = pickle.dumps(client_model_weights)
    client_socket.send(client_weights_bytes)
    client_socket.close()

@app.route('/client1', methods=['GET', 'POST'])
def client1():
    
    if(validation2==True):
        t = threading.Thread(target=client_worker, args=(model1, 1))
        t.start()
        #return "Client 1 connected."
        return render_template("intimation.html", n="Client 1 Transfered Data Successfully")
    else:
        msg="Not Valid User"
        return render_template('error.html',n=msg)

        

@app.route('/client2', methods=['GET', 'POST'])
def client2():
    if(validation3==True):
        
        t = threading.Thread(target=client_worker, args=(model2, 2))
        t.start()
        #return "Client 2 connected."
        return render_template("intimation.html", n="Client 2 Transfered Data Successfully")

    else:
        msg="Not Valid User"
        return render_template('error.html',n=msg)


@app.route('/client3', methods=['GET', 'POST'])
def client3():

    if(validation4==True):
        
        t = threading.Thread(target=client_worker, args=(model3, 3))
        t.start()
        #return "Client 3 connected."
        return render_template("intimation.html", n="Client 3 Transfered Data Successfully")

    else:
        msg="Not Valid User"
        return render_template('error.html',n=msg)


##########################################***************************************************************########################################
#REVERSE PROCESS MEANS FROM GLOBAL SERVER TO LOCAL SERVERS#

client_number=0

@app.route('/client_as_server', methods=['GET', 'POST'])
def server_active2():

    global client_number

    if(validation2==True and validation3==True and validation4==True):
        # Define the server's IP address and port
        server_ip = '10.10.16.114'  # Replace with your server's local IP address
        server_port = 12347

        # Create a server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the server socket to the IP address and port
        server_socket.bind((server_ip, server_port))

        # Listen for incoming connections
        server_socket.listen(1)


        # Accept connections from multiple clients
        num_rounds = 1
        num_clients = 1
        client_weights_list = []

        

        for _ in range(num_clients):
            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")

            client_weights_bytes = b''
            for round_num in range(num_rounds):
                while True:
                    data = client_socket.recv(238164608)  # Adjust buffer size as needed
                    if not data:
                        break
                    client_weights_bytes += data

                if not client_weights_bytes:
                    print(f"No data received from a client{_ + 1}, in round {round_num}.")
                    continue

                try:
                    client_model_weights = pickle.loads(client_weights_bytes)

                    #global client_number

                    if(client_number==1):
                        
                        #model1.set_weights(client_model_weights)
                        model1.save("local_model1_updated.keras")

                    elif (client_number==2):
                        #model2.set_weights(client_model_weights)
                        model2.save("local_model2_updated.keras")

                    elif(client_number==3):
                        #model3.set_weights(client_model_weights)
                        model3.save("local_model3_updated.keras")

                    else:
                        print("sorry I couldn't update model")
                        return render_template("welcome.html")
            
                except pickle.UnpicklingError as e:
                    print(f"Error while unpickling data from a client: {e}")  

            client_socket.close()

        return render_template("welcome.html")

    else:
        msg="Not Valid User"
        return render_template('error.html',n=msg)


def client_worker2(model, client_num):
    server_ip = '10.10.16.114'  # Replace with your server's local IP address
    server_port = 12347
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    global_model_weights = model.get_weights()
    global_weights_bytes = pickle.dumps(global_model_weights)

    client_socket.send(global_weights_bytes)
    client_socket.close()

@app.route('/server_as_client' , methods=['GET', 'POST'])
def sclient1():    
    global client_number
    client_number=client_number+1
    if(validation1==True and client_number<4):
        
        t = threading.Thread(target=client_worker, args=(global_model, client_number))
        t.start()
        #return "Client 1 connected."
        return render_template("intimation.html", n="Server Transfered Data Successfully To Client")
    
    else:
        msg="Not Valid User"
        return render_template('error.html',n=msg)

if __name__ == '__main__':
    app.run(debug=True)
