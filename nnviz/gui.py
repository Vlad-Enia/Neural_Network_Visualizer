import PySimpleGUI as sg
from PIL import Image

WIDTH = 1600
HEIGHT = 900

def define_layout():
    layout = [
        [sg.Frame('', [[sg.Image(key='-IMAGE-', expand_x=True, expand_y=True, size=(WIDTH, HEIGHT-200), background_color='white')]], size=(WIDTH, HEIGHT-200), element_justification='c', background_color='white')],
        [
            sg.Frame('Layer Size', element_justification='r',
                     layout=[
                        [sg.Text('Input layer'), sg.In(size=(2, 1), key='-INPUT_LAYER_SIZE-', default_text='2')],
                        [sg.Text('Hidden layer'), sg.In(size=(2, 1), key='-HIDDEN_LAYER_SIZE-', default_text='6')],
                        [sg.Text('Output layer'), sg.In(size=(2, 1), key='-OUTPUT_LAYER_SIZE-', default_text='1')]]),

            sg.Frame('Activation, Loss and Optimizer', element_justification='r',
                     layout=[
                         [sg.Text('Hidden layer activation'), sg.Combo(values=('Linear', 'Sigmoid', 'Relu'), default_value='Linear', key='-HIDDEN_ACTIVATION-')],
                         [sg.Text('Output layer activation'), sg.Combo(values=('Linear', 'Sigmoid', 'Relu'), default_value='Linear', key='-OUTPUT_ACTIVATION-')],
                         [sg.Text('Loss function'),sg.Combo(values=('binary_crossentropy', 'mean_squared_error', 'huber'), default_value='binary_crossentropy',key='-LOSS-')],
                         [sg.Text('Optimizer                        '), sg.Combo(values=('adam', 'SGD', 'RMSprop'),default_value='RMSprop', key='-OPT-')],
                         [sg.Button('Draw Graph', enable_events=True, key='-RENDER-')]]),

            sg.Frame('Training', element_justification='r',
                     layout=[
                        # [sg.Text('Learning rate'), sg.In(size=(5, 1), key='-LEARNING_RATE-', default_text='0.01')],
                        [sg.Text('Number of epochs'), sg.Combo(values=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], default_value=1000, key='-EPOCHS-')],
                        [sg.Text('Animation frequency'), sg.Combo(values=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500], default_value=200, key='-FREQ-')],
                        [sg.Button('Train', enable_events=True, key='-TRAIN-', disabled=True, button_color='grey')]]),

            sg.Frame('Predict', element_justification='c',
                     layout=[
                         [sg.Text('Current epoch'), sg.Combo(values=[], key='-CURRENT_EPOCH-', enable_events=True, disabled=True)],
                         [sg.Button('Feed Forward', enable_events=True, key='-PREDICT-', disabled=True, button_color='grey')]]),

            sg.Frame('Animation Controls', element_justification='c',
                     layout=[
                         [sg.Button('Play Animation', enable_events=True, key='-PLAY_PAUSE-', button_color='grey', disabled=True)],
                         [sg.Button('Previous', enable_events=True, key='-PREV-', button_color='grey', disabled=True), sg.Button('Next', enable_events=True, key='-NEXT-', button_color='grey', disabled=True), ],])
        ],
    ]
    return layout


def render_gui(width=WIDTH, height=HEIGHT):
    window = sg.Window('Neural Network Visualizer', layout=define_layout(), size=(width, height), keep_on_top=True)
    return window


def check_params(values): #TODO check params for size = 1 for different layers
    message = ''
    layer_size_list = ['-INPUT_LAYER_SIZE-', '-HIDDEN_LAYER_SIZE-', '-OUTPUT_LAYER_SIZE-']
    for layer_size in layer_size_list:
        layer = layer_size[1:-1].split('_')[0].lower()
        try:
            int(values[layer_size])
        except ValueError:
            message += '\n' + f'Input field for {layer} layer size must contain an integer value!'
        else:
            if not values[layer_size]:
                msg = f'Input field for {layer} layer size is empty!'
                message += '\n' + msg
            elif int(values[layer_size]) < 1:
                msg = f'The neural network needs to have at least 1 neuron on the {layer} layer!'
                message += '\n' + msg
            elif int(values[layer_size]) > 30:
                msg = f'Too many neurons on the {layer} layer. The limit is 30!'
                message += '\n' + msg

    # try:
    #     float(values['-LEARNING_RATE-'])
    # except ValueError:
    #     message += '\n' + 'Input field for learning rate value must contain a float value!'
    # else:
    #     if not values['-LEARNING_RATE-']:
    #         msg = 'Input field for learning rate value is empty!'
    #         message += '\n' + msg
    #     elif float(values['-LEARNING_RATE-']) <= 0 or float(values['-LEARNING_RATE-']) >= 1:
    #         msg = 'Invalid learning rate value. It should be between 0 and 1!'
    #         message += '\n' + msg

    try:
        int(values['-EPOCHS-'])
    except ValueError:
        message += '\n' + 'Input the number of epochs must contain a float value!'
    else:
        if not values['-EPOCHS-']:
            msg = 'Input field for the number of epochs is empty!'
            message += '\n' + msg
        elif int(values['-EPOCHS-']) < 1:
            msg = 'At least one epoch should take place!'
            message += '\n' + msg

    return message

