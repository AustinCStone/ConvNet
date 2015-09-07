import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

srng = RandomStreams()

def custom_float(inp):
	# need to specify what type of float or will throw errors on some machines
	return np.asarray(inp, dtype=theano.config.floatX)


def init_weights(shape, initial_max_val=0.01):
	return theano.shared(custom_float(np.random.randn(*shape) * initial_max_val))


# theano really should have this built in
def relu(inp):
	return T.maximum(inp, 0.)


def dropout(inp, drop_prob=0.):
	if drop_prob > 0:
		retain_prob = 1 - drop_prob
		inp *= srng.binomial(inp.shape, p=retain_prob, dtype=theano.config.floatX)
		inp /= retain_prob
	return inp


def rms_prop(cost, params, accs, lr=0.001, rho=0.9, epsilon=1e-6):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for (grad, param, acc) in zip(grads, params, accs):
		acc_new = rho * acc + (1.0 - rho) * grad
		# epsilon is important here to avoid divide by zero
		grad_scaling = T.sqrt(acc_new ** 2 + epsilon)
		updates.append((param, param - lr * (grad / grad_scaling)))
		updates.append((acc, acc_new))
	return updates


def forward_prop(inp, w, w2, w3, w_o, p_drop_conv, p_drop_hidden):
	act_1 = relu(conv2d(inp, w, border_mode='full'))
	act_1_pooled = max_pool_2d(act_1, (2, 2))
	act_1_pooled = dropout(act_1_pooled, p_drop_conv)

	act_2 = relu(conv2d(act_1_pooled, w2))
	act_2_pooled = max_pool_2d(act_2, (2, 2))
	act_2_flattened = T.flatten(act_2_pooled, outdim=2)
	act_2_flattened = dropout(act_2_flattened, p_drop_conv)

	act_3 = relu(T.dot(act_2_flattened, w3))
	act_3 = dropout(act_3, p_drop_hidden)

	net_output = T.nnet.softmax(T.dot(act_3, w_o))
	return act_1_pooled, act_2_flattened, act_3, net_output


def run_net(num_iters, batch_size):
	train_input, test_input, train_output, test_output = mnist(onehot=True)

	# reshape image vectors into 4 tensor format for convolution
	train_input = train_input.reshape(-1, 1, 28, 28)
	test_input = test_input.reshape(-1, 1, 28, 28)

	input_sym = T.ftensor4()
	output_sym = T.fmatrix()

	w = init_weights((8, 1, 5, 5)) # 8 filters of size 5 X 5
	w2 = init_weights((8, 8, 3, 3)) # 8 filters of size 3 X 3
	w3 = init_weights((392, 625)) # fully connected layer
	w_o = init_weights((625, 10)) # fully connected output layer

	# accumulator variables for RMS prop
	acc_w = init_weights((8, 1, 5, 5))
	acc_w2 = init_weights((8, 8, 3, 3))
	acc_w3 = init_weights((392, 625))
	acc_wo = init_weights((625, 10))

	noise_act_1_pooled, noise_act_2_flattened, noise_l4, noise_net_output = \
		forward_prop(input_sym, w, w2, w3, w_o, 0.2, 0.5)
	act_1_pooled, act_2_flattened, l4, net_output = \
		forward_prop(input_sym, w, w2, w3, w_o, 0., 0.)
	prediction = T.argmax(net_output, axis=1)

	cost = T.mean(T.nnet.categorical_crossentropy(noise_net_output, output_sym))
	params = [w, w2, w3, w_o]
	accs = [acc_w, acc_w2, acc_w3, acc_wo]
	updates = rms_prop(cost, params, accs, lr=0.001)

	train = theano.function(inputs=[input_sym, output_sym], outputs=cost, updates=updates, \
		allow_input_downcast=True)
	predict = theano.function(inputs=[input_sym], outputs=prediction, allow_input_downcast=True)

	for i in range(num_iters):
		for batch_start in range(0, len(train_input) - batch_size, batch_size):
			cost = train(train_input[batch_start:batch_start + batch_size], \
				train_output[batch_start:batch_start + batch_size])
		print np.mean(np.argmax(test_output, axis=1) == predict(test_input))


if __name__ == "__main__":
    run_net(100, 128)
