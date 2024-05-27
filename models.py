import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        # "*** YOUR CODE HERE ***"
        w = self.get_weights()
        return nn.DotProduct(w, x)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        #"*** YOUR CODE HERE ***"
        result = self.run(x)
        #print("valor result", result)
        result_as_escalar = nn.as_scalar(result)
        if result_as_escalar >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """

        "*** YOUR CODE HERE ***"
        while True:
            mistakes_made = False
            
            # Loop over the dataset
            for x, y in dataset.iterate_once(1):

                prediction = self.get_prediction(x)
                #multiplier is 1 if he predict a wrong -1 or multiplier is -1 if he predict a wrong 1
                if prediction != nn.as_scalar(y):
                    self.get_weights().update(x, nn.as_scalar(y))
                    mistakes_made = True
            
            if not mistakes_made:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        size_1 = 1
        size_2 = 128
        size_3 = 64
        self.w1 = nn.Parameter(size_1, size_2)
        self.b1 = nn.Parameter(size_1, size_2)
        self.w2 = nn.Parameter(size_2, size_3)
        self.b2 = nn.Parameter(size_1, size_3)
        self.w3 = nn.Parameter(size_3, size_1)
        self.b3 = nn.Parameter(size_1, size_1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        bias1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        hidden1 = nn.ReLU(bias1)
        bias2 = nn.AddBias(nn.Linear(hidden1, self.w2), self.b2)
        hidden2 = nn.ReLU(bias2)
        output = nn.AddBias(nn.Linear(hidden2, self.w3), self.b3)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        loss = nn.SquareLoss(y_hat, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lr= 0.012
        num_epochs=550
        for epoch in range(num_epochs):
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                #if nn.as_scalar(loss) <= 0.15:
                w1_grad, b1_grad, w2_grad, b2_grad, w3_grad, b3_grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(w1_grad, -lr)
                self.b1.update(b1_grad, -lr)
                self.w2.update(w2_grad, -lr)
                self.b2.update(b2_grad, -lr)
                self.w3.update(w3_grad, -lr)
                self.b3.update(b3_grad, -lr)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        size_1 = 784 # input
        size_2 = 64  # hidden layer 1
        size_3 = 32   # hidden layer 2
        size_5 = 10  # output
        self.w1 = nn.Parameter(size_1, size_2)
        self.b1 = nn.Parameter(1, size_2)
        self.w2 = nn.Parameter(size_2, size_3)
        self.b2 = nn.Parameter(1, size_3)
        self.w3 = nn.Parameter(size_3, size_5)
        self.b3 = nn.Parameter(1, size_5)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        bias1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        hidden1 = nn.ReLU(bias1)
        bias2 = nn.AddBias(nn.Linear(hidden1, self.w2), self.b2)
        hidden2 = nn.ReLU(bias2)
        output = nn.AddBias(nn.Linear(hidden2, self.w3), self.b3)
        return output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        loss = nn.SoftmaxLoss(y_hat, y)
        return loss
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lr= 1e-2
        acc = 0
        while acc <= 0.975:
            
            for x, y in dataset.iterate_once(20):    
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(grad[0], -lr)
                self.b1.update(grad[1], -lr)
                self.w2.update(grad[2], -lr)
                self.b2.update(grad[3], -lr)
                self.w3.update(grad[4], -lr)
                self.b3.update(grad[5], -lr)
            acc = dataset.get_validation_accuracy()
            print("acc: ", acc)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        size_1 = self.num_chars # input
        size_2 = 96  # hidden layer 1
        size_5 = len(self.languages)  # output
        self.w1 = nn.Parameter(size_1, size_2)
        self.b1 = nn.Parameter(1, size_2)
        self.w2 = nn.Parameter(size_2, size_2)
        self.b2 = nn.Parameter(1, size_2)
        self.w3 = nn.Parameter(size_2, size_2)
        self.b3 = nn.Parameter(1, size_2)
        self.w4 = nn.Parameter(size_2, size_5)
        self.b4 = nn.Parameter(1, size_5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h1 = nn.AddBias(nn.Linear(xs[0], self.w1), self.b1)
        h_total = nn.ReLU(h1)

        for x in xs[1:]:
            hi = nn.AddBias(nn.Linear(x, self.w1), self.b1)

            h_total = nn.Add(nn.Linear(h_total, self.w2), nn.Linear(hi, self.w2))

        h2 = nn.AddBias(nn.Linear(h_total, self.w3), self.b3)  # Activation of the first hidden layer
        h2 = nn.ReLU(h2)      
        output = nn.AddBias(nn.Linear(h2, self.w4), self.b4)
        
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(xs)
        loss = nn.SoftmaxLoss(y_hat, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lr= 1e-2
        acc = 0
        while acc <= 0.845:
            acc = dataset.get_validation_accuracy()
            for x, y in dataset.iterate_once(20):    
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4])
                self.w1.update(grad[0], -lr)
                self.b1.update(grad[1], -lr)
                self.w2.update(grad[2], -lr)
                self.b2.update(grad[3], -lr)
                self.w3.update(grad[4], -lr)
                self.b3.update(grad[5], -lr)
                self.w4.update(grad[6], -lr)
                self.b4.update(grad[7], -lr)