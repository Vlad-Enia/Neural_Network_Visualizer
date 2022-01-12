# Neural Network Visualizer - AI Course Final Project

Neural Network Visualizer is a graphical tool for designing and training simple neural networks with only one hidden layer.


## Network Visualization

The app offers a simple GUI for choosing the number of neurons on each layer. After choosing the activation function for each layer, the loss function and the optimizer, 
by simply pressing the 'Draw Graph' button, a graph correspoding to the the neural network is rendered on screen:

<p align="center">
    <img src="https://github.com/Vlad-Enia/Neural_Network_Visualizer/blob/master/readme_images/nn_render.gif"/>
</p>


## Training

After drawing the graph and, at the same time, creating the model, the training controls become available. Here, a number of training epochs can be selected. 
Another parameter that can be set in the training section is 'Animation Frequency'. This parameter's value specifies after how many epochs an animation should be created.

For example, if we train for 700 epochs, we can set the animation frequency to 350, meaning that 3 animations will be created:
  - at epoch 0, before any training
  - at epoch 350, halfway through the training session
  - at epoch 700, meaning after the training is done
  
Pressing the 'Train' button will start the training session. We know the training is done when 'Current epoch' dropdown menu, from the 'Predict' section, becomes available:

<p align="center">
    <img src="https://github.com/Vlad-Enia/Neural_Network_Visualizer/blob/master/readme_images/nn_train.gif"/>
</p>

For the purpose of this presentation, a simple problem was chosen as an example: the XOR function. So we define our X (train set) and Y (labels) as follows:

```python
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

  Y = np.array([x[0] ^ x[1] for x in X])
```


## Prediction and Feed-Forward Animation

After training the neural network, we can check how the neural network predicts the input set by viewing the generated animations. By following the example from above, 
we can choose from 3 avaible animations, at epoch 0, 350 and 700, so before, halfway and after the training session. Therefore, these values will be available to choose
from the 'Current epoch' dropdown menu.

After choosing a value for the current epoch, we can press the button 'Feed Forward' and input values will appear on the input layer:

<p align="center">
    <img src="https://github.com/Vlad-Enia/Neural_Network_Visualizer/blob/master/readme_images/nn_feed_forward.gif"/>
</p>

After that, we can either
  - Play/Pause the animation:
  
<p align="center">
    <img src="https://github.com/Vlad-Enia/Neural_Network_Visualizer/blob/master/readme_images/auto_animation.gif"/>
</p>

  - Manually iterate through the animation frames:
  
<p align="center">
    <img src="https://github.com/Vlad-Enia/Neural_Network_Visualizer/blob/master/readme_images/manual_animation.gif"/>
</p>

The generated animations are pretty detailed, as we can see how an input is fed through the network:
  - input values on the input layer
  - weights "between" the input and the hidden layer
  - all the values of the actiovation function on each neuron from the hidden layer
  - weights "between" the hidden and the output layer
  - scores on the output layer

We can see that, after 700 training epochs, for (0, 0) and (1, 1) the neural network returns scores that can be round to 0 (aprox. 0.382 and 0.385) and for (0, 1) and (1, 0) we receive values
that round up to 1 (aprox. 0.619 and 0.616), which are expected results for the XOR function, meaning that our neural network succesfully learned the function.

