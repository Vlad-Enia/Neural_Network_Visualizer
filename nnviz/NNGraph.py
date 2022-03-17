import shutil
import os
from PIL import Image as im
from networkx.drawing.nx_agraph import to_agraph
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from networkx import DiGraph, set_node_attributes, set_edge_attributes
import pygraphviz
from pyvis.network import Network
from IPython.core.display import display, HTML
from pprint import pprint


# TODO: rezolva problema cu pygraphviz/graphviz etc

class NNGraph():
    def __init__(self, model_params, model_nb):
        self.model = self.create_model(model_params)
        self.graph = self.create_graph(self.model)
        self.graph_init = self.graph.copy()
        self.partial_models = self.get_partial_models()
        self.model_nb = model_nb
        if os.path.isdir(f'./model_{model_nb}_predictions'):
            shutil.rmtree(f'./model_{model_nb}_predictions')
        os.mkdir(f'./model_{model_nb}_predictions')

    def create_model(self, model_params):
        """
        Creates a keras neural network model shaped by the given parameters stored in model_params.
        :param model_params:  a dictionary containing values for input, hidden and output layer size and activation function, learning rate and number of epochs; should contain the following keys: 'input_size', 'hidden_size', 'output_size', 'input act', 'hidden_act', 'output_act', 'learning_rate', 'epochs'
        :return: a compiled keras model, with loss function mean squared error, and optimizer SGD
        """
        model = Sequential()
        model.add(Dense(model_params['hidden_size'], activation=model_params['hidden_act'], input_shape=(model_params['input_size'],)))
        model.add(Dense(model_params['output_size'], activation=model_params['output_act']))
        model.compile(loss=model_params['loss'], optimizer=model_params['optimizer'])
        return model

    def create_graph(self, model):
        """
        This function creates networkx digraph, representing a compiled neural network keras model given as parameter.
        :param model: compiled keras sequential neural network moedel
        :return: networkx digraph resembling the model
        """
        graph = DiGraph()
        for l in range(len(model.layers)):
            layer = model.layers[l]

            for n in range(0, layer.input_shape[1]):
                if l == 0:
                    graph.add_node(
                        f'{l}_{n}',
                        shape='circle',
                        color='lightblue',
                        label='',
                        width=0.6,
                        fixedsize=True
                    )
                else:
                    graph.add_node(
                        f'{l}_{n}',
                        shape='circle',
                        color='lightgreen',
                        label='',
                        width=0.6,
                        fixedsize=True
                    )

        l = len(model.layers) - 1
        layer = model.layers[l]
        for n in range(0, layer.output_shape[1]):
            graph.add_node(
                f'{l + 1}_{n}',
                shape='ellipse',
                color='lightblue',
                label='',
                width=1.5,
                fixedsize=True
            )

        for l in range(len(model.layers)):
            layer = self.model.layers[l]
            for n in range(0, layer.input_shape[1]):
                for o in range(0, layer.output_shape[1]):
                    graph.add_edge(
                        f'{l}_{n}',
                        f'{l + 1}_{o}',
                        color='black',
                    )
        return graph

    def render_graph(self, graph, filename):
        """
        Exports a networkx digraph given as parameter to a .png file with name = filename given as parameter.
        :param graph: networkx digraph that will be exported
        :param filename: name of the .png file that will contain the drawing of the graph
        """
        out = to_agraph(graph)
        out.layout(prog='dot')
        out.draw(filename + '.png')
        return im.open(filename + '.png')

    def render_pyvis_graph(self):
        g = Network(height='400px', width='50%',heading='')
        g.from_nx(self.graph)
        g.show_buttons(filter_=['physics'])
        g.show('neural_network.html')
        display(HTML('neural_network.html'))

    def get_partial_models(self):
        """
        Creates a list of partial models of the self.model.
        A partial model contains all the layers of a model (with matching weights and activations) up to a specified layer depth.
        These partial models are used for animating neuron activations.
        Example: if a model contains three layers (input, hidden and output layer), there would be two partial models:
        - one with only the input layer
        - one containing the input and the hidden layer.
        :return: a list containing all the partial model of the self.model
        """

        part_models = []
        for i in range(len(self.model.layers)):
            part_model = Sequential()
            for j in range(i + 1):
                prev_layer = self.model.layers[j]
                part_model.add(Dense(
                    prev_layer.output_shape[1],
                    input_dim=prev_layer.input_shape[1],
                    activation=prev_layer.activation
                ))
                part_model.layers[j].set_weights(prev_layer.get_weights())
            part_model.compile(loss=self.model.loss, optimizer=self.model.optimizer)
            part_models.append(part_model)
        return part_models

    def reset_graph(self):
        self.graph = self.graph_init.copy()

    def create_gif(self, img_list, filename):
        img_list[0].save(
            f'{filename}.gif',
            optimize=False,
            save_all=True,
            append_images=img_list[1:],
            loop=0,
            duration=1000
        )

    def animate_predictions(self, x, epoch):
        """
        Feeds input data to the partial models and to the whole model, and exports graphs at every step (activation on layer l, followed by weights between layer l and l+1 and so on).
        This method is called after 'epoch' number of training epochs, so that we can observe the network's evolution
        :param x: input data
        :param epoch: number of training epochs that took place before calling this method
        """
        graph_images = []
        predictions = [x]

        for i in range(len(self.partial_models)):
            predictions.append(self.partial_models[i].predict(x))
        predictions.append(self.model.predict(x))

        dir_path = f'./model_{self.model_nb}_predictions/epoch_{epoch}'
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)

        k = 0   # used to order images

        for i in range(len(x)):
            for l in range(len(self.model.layers)):
                layer = self.model.layers[l]

                for n in range(0, layer.input_shape[1]):
                    act = "{:.2f}".format(predictions[l][i][n])

                    index = f'{l}_{n}'

                    if l == 0:
                        set_node_attributes(self.graph, {
                            index: {
                                'style': 'filled',
                                'color': 'lightblue',
                                'label': act
                            }})
                    else:
                        set_node_attributes(self.graph, {
                            index: {
                                'style': 'filled',
                                'color': 'lightgreen',
                                'label': act
                            }})

                graph_images.append(self.render_graph(self.graph, f'{dir_path}/{k}_act_{i}_{l}'))
                # self.reset_graph()
                k += 1

                weights = self.model.layers[l].get_weights()[0]     # [0] for only the weights

                for n in range(layer.input_shape[1]):
                    for o in range(layer.output_shape[1]):
                        index = (f'{l}_{n}', f'{l+1}_{o}')
                        weight = "{:.2f}".format(weights[n][o])

                        set_edge_attributes(self.graph, {
                            index: {
                                'label': weight
                            }})

                graph_images.append(self.render_graph(self.graph, f'{dir_path}/{k}_weights_{i}_{l}_{l+1}'))
                self.reset_graph()

                k += 1

                if l == len(self.model.layers) - 1:
                    for h in range(0, layer.output_shape[1]):
                        act = predictions[l+1][i][h]

                        index = f'{l+1}_{h}'

                        set_node_attributes(self.graph, {
                            index: {
                                'style': 'filled',
                                'color': 'lightblue',
                                'label': act
                            }})
                    graph_images.append(self.render_graph(self.graph, f'{dir_path}/{k}_act_{i}_{l+1}'))
                    self.reset_graph()
                    k += 1

        # self.create_gif(graph_images, 'graph_predictions')
        # return graph_images

    def train(self, x, y, epochs, epoch_freq):
        n = len(x)
        e = 0
        self.animate_predictions(x, e)
        while e < epochs:
            e += epoch_freq
            self.model.fit(x, y, batch_size=n, epochs=epoch_freq, verbose=0)
            self.partial_models = self.get_partial_models()
            self.animate_predictions(x, e)
            print(f'first {e} epochs done')