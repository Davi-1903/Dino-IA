from numpy import exp, log, tanh, dot, array, random, sqrt, int64, float64
from typing import Literal, Callable
from random import gauss


class Neuron:
    '''Neurônio matemático.
    
    Atributos:
    - `value:` Propriedade que guarda o valor recebido (privado);
    - `bias:` Viés cognitivo;
    - `activation_function:` Função de ativação;
    '''
    __slots__ = ['__value', '__bias', 'activation_function']

    def __init__(self, activation_function: Literal['none', 'sigmoid', 'swish', 'tanh', 'relu', 'leaky_relu', 'softplus']):
        '''Método construtor.
        
        Parâmetros:
        - `activation_function:` Função de ativação;
            - none;
            - sigmoid;
            - swish;
            - tanh;
            - relu;
            - leaky relu;
            - softplus;
        '''
        self.__value = 0
        self.__bias = 0
        self.activation_function = self.select_activation_function(activation_function)

    def select_activation_function(self, name: str) -> Callable[[float], float]:
        '''Método que retorna a função de ativação.
        
        Parâmetros:
        - `name:` Nome da função de ativação;
        '''
        functions = {
            'none': lambda x: x - self.__bias,
            'sigmoid': lambda x: 1 / (1 + exp(-x)),
            'swish': lambda x: x / (1 + exp(-x)),
            'tanh': tanh,
            'relu': lambda x: max(0, x),
            'leaky_relu': lambda x: max(0.01 * x, x),
            'softplus': lambda x: log(1 + exp(x))
        }
        if name not in functions:
            raise ValueError(f'Invalid name "{name}". Try: {", ".join(functions)}')
        return functions[name]

    @property
    def value(self) -> float:
        return self.activation_function(self.__value + self.__bias)

    @value.setter
    def value(self, raw_value: float):
        if not isinstance(raw_value, (int, float, int64, float64)):
            raise TypeError('The value must be a number.')
        self.__value = raw_value
    
    @property
    def bias(self) -> float:
        return self.__bias

    @bias.setter
    def bias(self, raw_bias: float):
        if not isinstance(raw_bias, (int, float, int64, float64)):
            raise TypeError('The bias must be a number.')
        self.__bias = raw_bias

    def get_raw(self) -> float:
        '''Método que retorna o valor sem viés e sem função de ativação.'''
        return self.__value


class NeuronRecurrent(Neuron):
    '''Neurônio recorrente matemático.
    
    Atributos:
    - `hidden_state:` Estado oculto do neurônio (privado);
    - `hidden_weights:` Pesos do estado oculto (privado);
    '''
    __slots__ = ['__hidden_state', '__hidden_weights']

    def __init__(self, activation_function: Literal['none', 'sigmoid', 'swish', 'tanh', 'relu', 'leaky_relu', 'softplus']):
        '''Método construtor.
        
        Parâmetros:
        - `activation_function:` Função de ativação;
            - none;
            - sigmoid;
            - swish;
            - tanh;
            - relu;
            - leaky relu;
            - softplus;
        '''
        super().__init__(activation_function)
        self.__hidden_state = 0
        self.__hidden_weights = gauss(0, 1)

    @Neuron.value.getter
    def value(self) -> float:
        '''Método que retorna o valor do neurônio após aplicar o estado oculto, pesos e função de ativação.'''
        raw_value = self.get_raw() + self.__hidden_state * self.__hidden_weights + self.bias
        self.__hidden_state = self.activation_function(raw_value)
        return self.__hidden_state

    @property
    def hidden_weights(self) -> float:
        '''Método que retorna os pesos do estado oculto.'''
        return self.__hidden_weights

    @hidden_weights.setter
    def hidden_weights(self, raw_hidden_weights: float):
        '''Método para definir os pesos do estado oculto.
        
        Parâmetros:
        - `raw_hidden_weights:` Valor dos pesos;
        '''
        if not isinstance(raw_hidden_weights, (int, float, int64, float64)):
            raise TypeError('The hidden weights must be a number.')
        self.__hidden_weights = raw_hidden_weights

    def reset_hidden_state(self):
        '''Método para reiniciar o estado oculto.'''
        self.__hidden_state = 0


class Layer:
    '''Camada de neurônios.
    
    Atributos:
    - `neurons:` Lista que guarda os neurônios (privado);
    '''
    __slots__ = ['__neurons']

    def __init__(self, numbers_of_neurons: int, activation_function: Literal['none', 'sigmoid', 'swish', 'tanh', 'relu', 'leaky_relu', 'softplus'], recurrent: bool = False):
        '''Método construtor.
        
        Parâmetros:
        - `numbers_of_neurons:` Quantidade de neurônios;
        - `activation_function:` Função de ativação;
        - `recurrent:` Se o neurônio é recorrente (opcional);

        ```
        >>> Layer(5, 'sigmoid')
        >>> Layer(5, 'sigmoid', True)
        ```
        '''
        self.__neurons = [NeuronRecurrent(activation_function) if recurrent else Neuron(activation_function) for n in range(numbers_of_neurons)]

    def __len__(self) -> int:
        return len(self.__neurons)

    def get_biases(self) -> list:
        '''Método que retorna as bias dos neurônios.'''
        return [neuron.bias for neuron in self.__neurons]

    def get_values(self) -> list:
        '''Método que retorna os valores dos neurônios.'''
        return [neuron.value for neuron in self.__neurons]

    def get_neurons(self) -> list:
        '''Método que retorna os neurônios da camada.'''
        return self.__neurons

    def set_biases(self, biases: list):
        '''Método para definir as biases dos neurônios.
        
        Parâmetros:
        - `biases:` Lista de valores (bias);
        '''
        for idx, bias in enumerate(biases):
            self.__neurons[idx].bias = bias

    def set_values(self, values: list, weigths: array):
        '''Método para atribuir valor aos neurônios.
        
        Parâmetros:
        - `values:` Lista de valores da camada anterior ou da camada de entrada;
        - `weigths:` Matriz de pesos entre as camadas;
        '''
        for idx, value in enumerate(dot(array(values), weigths)):
            self.__neurons[idx].value = value
    
    def reset_hidden_states(self):
        '''Método para reiniciar os estados ocultos dos neurônios.'''
        for neuron in self.__neurons:
            if isinstance(neuron, NeuronRecurrent):
                neuron.reset_hidden_state()

    def get_hidden_weights(self) -> list:
        '''Método que retorna os pesos dos neurônios ocultos.'''
        return [neuron.hidden_weights for neuron in self.__neurons if isinstance(neuron, NeuronRecurrent)]

    def set_hidden_weights(self, weights: list):
        '''Método para definir os pesos dos neurônios ocultos.
        
        Parâmetros:
        - `weights:` Lista de pesos;
        '''
        for idx, weight in enumerate(weights):
            if isinstance(self.__neurons[idx], NeuronRecurrent):
                self.__neurons[idx].hidden_weights = weight


class Network:
    '''Rede neural genérica do tipo Feed Forward.

    Atributos:
    - `inputs:` Quantidade de entrada (privado);
    - `hiddens:` Lista com a estrutura das camadas ocultas (privado);
    - `outputs:` Lista com a estrutura da camada de saída (privado);
    - `layers:` Lista de camadas (privado);
    - `weights:` Pesos da rede neural (privado);
    '''
    __slots__ = ['__inputs', '__hiddens', '__outputs', '__layers', '__weights']

    def __init__(self, structure: list[dict], weights: list | None = None, weights_initialization: Literal['random', 'xavier', 'he', 'lecun'] = 'random', biases: list | None = None, hidden_weights: list | None = None):
        '''Método construtor.
        
        Parâmetros:
        - `structure:` Estrutura da rede neural, lista de dicionários com entradas e funções de ativação (Primeira camada apenas com a quantidade de entradas);

            ```
            >>> Network([{'neurons': 1}, {'neurons': 2, 'function': 'sigmoid'}])
            ```
        - `weights:` Lista de pesos da rede neural (opcional);
        - `weights_initialization:` Método de inicialização dos pesos (opcional);
            - `random`;
            - `xavier`;
            - `he`;
            - `lecun`;
        - `biases:` Lista de bias da rede neural (opcional);
        - `hidden_weights:` Lista de pesos dos neurônios ocultos (opcional);
        '''
        self.__inputs, *self.__hiddens, self.__outputs = structure
        self.__layers = []
        for idx in range(1, len(structure)):
            self.__layers.append(Layer(**structure[idx]))
        self.set_weights(weights, weights_initialization)
        self.set_biases(biases)
        self.set_hidden_weights(hidden_weights)

    def set_weights(self, weights: list, weights_initialization: Literal['random', 'xavier', 'he', 'lecun'] = 'random'):
        '''Método para a definição dos pesos ou, caso não haja pesos, inicialização.
        
        Parâmetros:
        - `weights:` Lista de pesos da rede neural;
        - `weights_initialization:` Método de inicialização dos pesos (opcional);
            - `random`;
            - `xavier`;
            - `he`;
            - `lecun`;
        '''
        self.__weights = weights
        if not weights:
            self.__weights = []
            if weights_initialization == 'random':
                self.inicialization_random()
            elif weights_initialization == 'xavier':
                self.inicialization_xavier()
            elif weights_initialization == 'he':
                self.inicialization_he()
            elif weights_initialization == 'lecun':
                self.inicialization_lecun()
            else:
                raise ValueError(f'Invalid weights inicialization "{weights_initialization}". Choose from random, xavier, he, lecun')
    
    def inicialization_random(self):
        '''Método para inicialização de pesos com o método random.'''
        self.__weights.append(random.uniform(-0.01, 0.01, size=(self.__inputs['numbers_of_neurons'], len(self.__layers[0]))))
        for idx in range(1, len(self.__layers)):
            self.__weights.append(random.uniform(-0.01, 0.01, size=(len(self.__layers[idx - 1]), len(self.__layers[idx]))))
    
    def inicialization_xavier(self):
        '''Método para inicialização de pesos com o método xavier.'''
        limit = sqrt(6 / (self.__inputs['numbers_of_neurons'] + len(self.__layers[0])))
        self.__weights.append(random.uniform(-limit, limit, size=(self.__inputs['numbers_of_neurons'], len(self.__layers[0]))))
        for idx in range(1, len(self.__layers)):
            limit = sqrt(6 / (len(self.__layers[idx - 1]) + len(self.__layers[idx])))
            self.__weights.append(random.uniform(-limit, limit, size=(len(self.__layers[idx - 1]), len(self.__layers[idx]))))
    
    def inicialization_he(self):
        '''Método para inicialização de pesos com o método he.'''
        self.__weights.append(random.randn(self.__inputs['numbers_of_neurons'], len(self.__layers[0])) * sqrt(2 / self.__inputs['numbers_of_neurons']))
        for idx in range(1, len(self.__layers)):
            self.__weights.append(random.randn(len(self.__layers[idx - 1]), len(self.__layers[idx])) * sqrt(2 / len(self.__layers[idx - 1])))

    def inicialization_lecun(self):
        '''Método para inicialização de pesos com método lecun.'''
        self.__weights.append(random.randn(self.__inputs['numbers_of_neurons'], len(self.__layers[0])) * sqrt(1 / self.__inputs['numbers_of_neurons']))
        for idx in range(1, len(self.__layers)):
            self.__weights.append(random.randn(len(self.__layers[idx - 1]), len(self.__layers[idx])) * sqrt(1 / len(self.__layers[idx - 1])))

    def set_biases(self, biases: list):
        '''Método para a definição dos bias da rede neural.

        Parâmetros:
        - `biases:` Lista de bias da rede neural;
        '''
        if biases:
            for idx, biases in enumerate(biases):
                self.__layers[idx].set_biases(biases)

    def get_biases(self) -> list:
        '''Método que retorna as biases da rede neural.'''
        return [layer.get_biases() for layer in self.__layers]

    def get_weights(self) -> list:
        '''Método que retorna os pesos da rede neural.'''
        return self.__weights
    
    def get_hidden_weights(self) -> list:
        '''Método que retorna os pesos dos neurônios ocultos da rede neural.'''
        return [layer.get_hidden_weights() for layer in self.__layers]
    
    def set_hidden_weights(self, hidden_weights: list):
        '''Método para definir os pesos dos neurônios ocultos da rede neural.
        
        Parâmetros:
        - `hidden_weights:` Lista de pesos dos neurônios ocultos;
        '''
        if hidden_weights:
            for idx, weights in enumerate(hidden_weights):
                self.__layers[idx].set_hidden_weights(weights)

    def get_layers(self) -> list:
        '''Método que retorna as camadas da rede neural.'''
        return self.__layers

    def backpropagation(self, inputs: list, targets: list, learning_rate: float):
        '''
        Método para realizar a retropropagação (backpropagation) da rede neural.

        Parâmetros:
        - `inputs:` Lista de entradas da rede neural;
        - `targets:` Lista de saídas esperadas;
        - `learning_rate:` Taxa de aprendizado;
        '''
        outputs = self.forward(inputs)

        output_errors = array(targets) - array(outputs)

        output_gradients = output_errors * array([self.derivative(neuron.get_raw(), neuron.activation_function) for neuron in self.__layers[-1].get_neurons()])

        hidden_values = array(self.__layers[-2].get_values()) if len(self.__layers) > 1 else array(inputs)
        self.__weights[-1] += learning_rate * dot(hidden_values.reshape(-1, 1), output_gradients.reshape(1, -1))

        for i, neuron in enumerate(self.__layers[-1].get_neurons()):
            neuron.bias += learning_rate * output_gradients[i]

        hidden_errors = output_gradients
        for i in range(len(self.__layers) - 2, -1, -1):
            hidden_errors = dot(hidden_errors, self.__weights[i + 1].T)

            hidden_gradients = hidden_errors * array([self.derivative(neuron.get_raw(), neuron.activation_function) for neuron in self.__layers[i].get_neurons()])

            previous_values = array(inputs) if i == 0 else array(self.__layers[i - 1].get_values())
            self.__weights[i] += learning_rate * dot(previous_values.reshape(-1, 1), hidden_gradients.reshape(1, -1))

            for j, neuron in enumerate(self.__layers[i].get_neurons()):
                neuron.bias += learning_rate * hidden_gradients[j]

    def derivative(self, x: float, activation_function: Callable[[float], float]) -> float:
        '''
        Calcula a derivada da função de ativação.

        Parâmetros:
        - `x:` Valor de entrada;
        - `activation_function:` Função de ativação;
        '''
        if activation_function.__name__ == '<lambda>':
            if activation_function(0) == 0.5:
                return activation_function(x) * (1 - activation_function(x))
            elif activation_function(0) == 0:
                return 1 if x > 0 else 0
            elif activation_function(1) == 1:
                return 1
            else:
                sigmoid_x = 1 / (1 + exp(-x))
                return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
        elif activation_function.__name__ == 'tanh':
            return 1 - tanh(x) ** 2
        elif activation_function.__name__ == 'softplus':
            return 1 / (1 + exp(-x))
        else:
            return 1 if x > 0 else 0.01

    def forward(self, parameters: list) -> list:
        '''Método que realiza a feedforward da rede neural.

        Parâmetros:
        - `parameters:` Parâmetros de entrada da rede neural;
        '''
        self.__layers[0].set_values(parameters, self.__weights[0])
        for idx in range(1, len(self.__layers)):
            self.__layers[idx].set_values(self.__layers[idx - 1].get_values(), self.__weights[idx])
        return self.__layers[-1].get_values()
