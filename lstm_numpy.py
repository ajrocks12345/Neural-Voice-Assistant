import numpy as np

def sigmoid(x):
    # Clip x to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

class LSTMRNN:
    """
    A pure NumPy implementation of an LSTM-RNN for sequence classification.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        # Concatenate hidden state and input: size = hidden_size + input_size
        z_size = hidden_size + input_size
        
        # Forget gate
        self.Wf = np.random.randn(hidden_size, z_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))
        
        # Input gate
        self.Wi = np.random.randn(hidden_size, z_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))
        
        # Cell candidate
        self.Wc = np.random.randn(hidden_size, z_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))
        
        # Output gate
        self.Wo = np.random.randn(hidden_size, z_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))
        
        # Output layer (Fully Connected)
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        Forward pass for a sequence of inputs.
        inputs: list of NumPy arrays (each representing a timestep vector of shape (input_size, 1))
        Returns outputs, hidden states, cell states, and caches for BPTT.
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        h_states, c_states, outputs, caches = [], [], [], []
        
        for x in inputs:
            z = np.row_stack((h, x))
            
            f = sigmoid(np.dot(self.Wf, z) + self.bf)
            i = sigmoid(np.dot(self.Wi, z) + self.bi)
            c_hat = tanh(np.dot(self.Wc, z) + self.bc)
            
            c = f * c + i * c_hat
            o = sigmoid(np.dot(self.Wo, z) + self.bo)
            h = o * tanh(c)
            
            y = np.dot(self.Wy, h) + self.by
            # Softmax
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)
            
            h_states.append(h)
            c_states.append(c)
            outputs.append(p)
            caches.append((z, f, i, c_hat, o, c))
            
        return outputs, h_states, c_states, caches

    def backward(self, inputs, targets, outputs, h_states, c_states, caches):
        """
        Backpropagation Through Time (BPTT).
        inputs: list of timesteps
        targets: target class index (we assume sequence classification at the final timestep)
        """
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)
        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        
        # Compute loss gradient for the last timestep
        p = outputs[-1]
        dy = np.copy(p)
        dy[targets] -= 1
        
        dWy += np.dot(dy, h_states[-1].T)
        dby += dy
        
        dh_next = np.dot(self.Wy.T, dy)
        
        for t in reversed(range(len(inputs))):
            z, f, i, c_hat, o, c_prev = caches[t]
            c_prev = caches[t-1][-1] if t > 0 else np.zeros((self.hidden_size, 1))
            
            dh = dh_next
            do = dh * tanh(c_states[t])
            do_input = dsigmoid(o) * do
            
            dc = dh * o * dtanh(tanh(c_states[t])) + dc_next
            
            dc_hat = dc * i
            dc_hat_input = dtanh(c_hat) * dc_hat
            
            di = dc * c_hat
            di_input = dsigmoid(i) * di
            
            df = dc * c_prev
            df_input = dsigmoid(f) * df
            
            # Gradients for weights and biases
            dWf += np.dot(df_input, z.T)
            dbf += df_input
            dWi += np.dot(di_input, z.T)
            dbi += di_input
            dWc += np.dot(dc_hat_input, z.T)
            dbc += dc_hat_input
            dWo += np.dot(do_input, z.T)
            dbo += do_input
            
            # Gradient wrt z
            dz = (np.dot(self.Wf.T, df_input)
                  + np.dot(self.Wi.T, di_input)
                  + np.dot(self.Wc.T, dc_hat_input)
                  + np.dot(self.Wo.T, do_input))
            
            dh_next = dz[:self.hidden_size, :]
            dc_next = f * dc
            
        # Gradient clipping to prevent exploding gradients
        for dparam in [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        # Update weights
        self.Wf -= self.learning_rate * dWf
        self.bf -= self.learning_rate * dbf
        self.Wi -= self.learning_rate * dWi
        self.bi -= self.learning_rate * dbi
        self.Wc -= self.learning_rate * dWc
        self.bc -= self.learning_rate * dbc
        self.Wo -= self.learning_rate * dWo
        self.bo -= self.learning_rate * dbo
        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby
        
        # Calculate cross-entropy loss
        loss = -np.log(p[targets, 0] + 1e-8)
        return loss

    def train(self, X_train, Y_train, epochs=100):
        """
        Train the model on a dataset.
        X_train: list of sequences (each sequence is a list of timesteps)
        Y_train: list of target indices
        """
        for epoch in range(epochs):
            total_loss = 0
            for x_seq, y_target in zip(X_train, Y_train):
                outputs, h_states, c_states, caches = self.forward(x_seq)
                loss = self.backward(x_seq, y_target, outputs, h_states, c_states, caches)
                total_loss += loss
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X_train):.4f}")

    def predict(self, x_seq):
        """Predict the class of a sequence"""
        outputs, _, _, _ = self.forward(x_seq)
        return np.argmax(outputs[-1])
