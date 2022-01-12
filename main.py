import os

import PySimpleGUI
import PySimpleGUI as sg
from nnviz import gui
from nnviz.NNGraph import NNGraph
import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64encode

TIMEOUT = 1000
BUTTON_COLOR_DEFAULT = '#283b5b'
BUTTON_COLOR_DISABLED = 'grey'

window = gui.render_gui()

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]])
Y = np.array([x[0] ^ x[1] for x in X])

gif_playing = False
nn_created = False

graph_img_list = []  # will contain all the images from a feed-forward animation
n = 0  # length of graph_img_list
i = 0  # current image index
epoch = 0  # current epoch
model_nb = -1  # current model

while True:
    event, values = window.read(timeout=TIMEOUT)

    if event == PySimpleGUI.TIMEOUT_EVENT and gif_playing is True and graph_img_list:
        i = (i + 1) % n
        window['-IMAGE-'].update(source=graph_img_list[i])

    elif event == sg.WIN_CLOSED:
        break

    elif event == '-RENDER-':
        warning_messages = gui.check_params(values)
        if warning_messages:
            sg.Popup(warning_messages)
        else:

            window['-TRAIN-'].update(disabled=False, button_color=BUTTON_COLOR_DEFAULT)
            window['-CURRENT_EPOCH-'].update(disabled=True)
            window['-PREDICT-'].update(disabled=True, button_color=BUTTON_COLOR_DISABLED)
            window['-PLAY_PAUSE-'].update(disabled=True, button_color=BUTTON_COLOR_DISABLED)
            window['-NEXT-'].update(disabled=True, button_color=BUTTON_COLOR_DISABLED)
            window['-PREV-'].update(disabled=True, button_color=BUTTON_COLOR_DISABLED)
            model_params = {
                'input_size': int(values['-INPUT_LAYER_SIZE-']),
                'hidden_size': int(values['-HIDDEN_LAYER_SIZE-']),
                'output_size': int(values['-OUTPUT_LAYER_SIZE-']),
                'hidden_act': values['-HIDDEN_ACTIVATION-'].lower(),
                'output_act': values['-OUTPUT_ACTIVATION-'].lower(),
                'loss': values['-LOSS-'],
                'optimizer': values['-OPT-']
                # 'learning_rate': float(values['-LEARNING_RATE-']),
            }
            model_nb += 1
            nn_graph = NNGraph(model_params, model_nb)
            nn_graph.render_graph(nn_graph.graph, 'graph')
            window['-IMAGE-'].update(source='./graph.png')
            # nn_graph.render_pyvis_graph()

    elif event == '-TRAIN-':

        epoch_frequency = int(values['-FREQ-'])
        epochs = int(values['-EPOCHS-'])
        e = 0
        available_epochs = [e]
        while e < int(epochs):
            e += epoch_frequency
            available_epochs.append(e)
        window['-CURRENT_EPOCH-'].update(disabled=False, values=available_epochs, size=(4, 5))

        nn_graph.train(X, Y, epochs, epoch_frequency)

    elif event == '-CURRENT_EPOCH-':
        window['-PREDICT-'].update(disabled=False, button_color=BUTTON_COLOR_DEFAULT)

    elif event == '-PREDICT-':
        current_epoch = values['-CURRENT_EPOCH-']
        graph_img_list = []
        dir_path = f'./model_{model_nb}_predictions/epoch_{current_epoch}'
        filename_list = []
        for filename in os.listdir(dir_path):
            filename_list.append(filename)
        filename_list.sort(key=lambda s: int(s.split('_')[0]))
        for filename in filename_list:
            f = os.path.join(dir_path, filename)
            img = Image.open(f)
            img_bytes = BytesIO()
            img.save(img_bytes, 'PNG')
            graph_img_list.append(img_bytes.getvalue())

        i = 0
        n = len(graph_img_list)
        window['-IMAGE-'].update(source=graph_img_list[i])

        window['-PLAY_PAUSE-'].update(disabled=False, button_color=BUTTON_COLOR_DEFAULT)
        window['-NEXT-'].update(disabled=False, button_color=BUTTON_COLOR_DEFAULT)
        window['-PREV-'].update(disabled=False, button_color=BUTTON_COLOR_DEFAULT)

    elif event == '-PLAY_PAUSE-':
        if gif_playing is True:
            gif_playing = False
            window['-PLAY_PAUSE-'].update('Play Animation')
            window['-NEXT-'].update(disabled=False, button_color=BUTTON_COLOR_DEFAULT)
            window['-PREV-'].update(disabled=False, button_color=BUTTON_COLOR_DEFAULT)
        else:
            gif_playing = True
            window['-PLAY_PAUSE-'].update('Pause Animation')
            window['-NEXT-'].update(disabled=True, button_color=BUTTON_COLOR_DISABLED)
            window['-PREV-'].update(disabled=True, button_color=BUTTON_COLOR_DISABLED)

    elif event == '-NEXT-':
        i = (i + 1) % n
        window['-IMAGE-'].update(source=graph_img_list[i])

    elif event == '-PREV-':
        i = (i - 1) % n
        window['-IMAGE-'].update(source=graph_img_list[i])
window.close()
