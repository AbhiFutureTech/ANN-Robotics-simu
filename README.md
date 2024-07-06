# Neural-Network-Based-Autonomous-Navigation-for-Self-Driving-Robots

1.The Robot uses neural network to learn and predict decisions just like a human brain.

2.The model is built using Neural Network and it's trained by feeding in labelled images of tracks.

3.After the model is trained it will be capable of making its own decisions. The prediction will be made on the laptop due to larger memory and flexibility. Raspberry pi will be used to stream the video to 
  laptop using Pi-camera.

4.First we will train the model using the dataset that contains the labelled images of the track.

5.Raspberry Pi will stream the live feed to the laptop and the predictions will be sent back to the raspberry pi.

6.The raspberry pi is connected to motor driver which will control the wheels of the bot. Ultrasonic sensor makes sure that the robot does not collide with obstacles. Once trained it can run autonomously and 
  make its decisions.It will try to maintain its path along the track and prevent from collision


## A) Hardware Design

The Hardware components used for this project are as follows:

1.Raspberry Pi (any model with sufficient performance, such as Raspberry Pi 4)
2.Camera module
3.Motor driver (e.g., L298N)
4.Motors and wheels
5.Power supply

## Required Libraries

Install the required libraries on your Raspberry Pi:

 ``` 
sudo apt-get update
sudo apt-get install python3-opencv
pip3 install tensorflow numpy picamera
 ```

![Demo (1)](https://github.com/patilabhi20/Robotic-Tasks-via-Large-Language-Models/assets/157373320/c83189c3-d478-4657-8c9e-ae332751a466)

## Installation

To set up the project, clone the repository and install the required dependencies:


 ``` 
git clone https://github.com/yourusername/Neural-Network-Based-Autonomous-Navigation-for-Self-Driving-Robots.git
cd Neural-Network-Based-Autonomous-Navigation-for-Self-Driving-Robots
pip install -r requirements.txt

 ```

Ensure you have the necessary hardware and software configurations as detailed in the documentation.

## B) Software Design

1.Python(2.7)
2.TensorFlow
3.OpenCV

## Model Architecture

The model architecture is based on a convolutional neural network (CNN) designed for real-time processing. The architecture includes convolutional layers for feature extraction and dense layers for decision making.


 ``` 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)  # Regression output for steering angle
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

 ``` 


## Training

The training pipeline includes data augmentation and preprocessing steps to enhance model performance. The model is trained using the mean squared error loss function and the Adam optimizer.

 ``` 

# Training the model
batch_size = 32
epochs = 10

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_steps=len(X_val) // batch_size
)

 ```

## Evaluation

Evaluate the model's performance using the validation set and visualize the results:


 ``` 
import matplotlib.pyplot as plt

# Evaluate the model
val_loss = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

 ```

## Examples

1.Object Recognition

 ```
from recognition.cnn_model import CNNModel

cnn_model = CNNModel()
cnn_model.load_model('models/object_recognition_model.h5')
result = cnn_model.predict('images/test_image.jpg')
print(f"Recognized Object: {result}")
Image Preprocessing

 ```

 ```
from processing.image_preprocessor import ImagePreprocessor

image_preprocessor = ImagePreprocessor()
preprocessed_image = image_preprocessor.process('images/raw_image.jpg')
preprocessed_image.show()

 ```

## 
Thank you for your interest in the Neural Network-Based Autonomous Navigation for Self-Driving Robots project! We hope you find it useful and engaging. Happy coding!
