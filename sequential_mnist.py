import numpy as np
import theano, theano.tensor as T
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes
import matplotlib.pyplot as plt
from matplotlib import cm


#### Helper Functions
def zeros(shape):
    return np.zeros(shape, dtype=theano.config.floatX)

def ones(shape):
    return np.ones(shape, dtype=theano.config.floatX)

def orthogonal(shape):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]].astype(theano.config.floatX)



### Dataset loader
_datasets = None
def get_dataset(which_set):
    global _datasets
    if not _datasets:
        MNIST = fuel.datasets.MNIST
        # jump through hoops to instantiate only once and only if needed
        _datasets = dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))
    return _datasets[which_set]

def get_stream(which_set, batch_size, num_examples=None):
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream


### Optimizer
def adam(lr, tparams, grads, inp, cost, extra, on_unused_input='warn'):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%p.name) for p in tparams]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra, updates=gsup, on_unused_input=on_unused_input)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (T.sqrt(fix2) / fix1)

    for p, g in zip(tparams, gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost, extra):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad'%p.name) for p in tparams]
    running_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad'%p.name) for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2'%p.name) for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, [cost]+extra,
                                    updates=zgup+rgup+rg2up, profile=False)

    updir = [theano.shared(p.get_value() * np.float32(0.), name='%s_updir'%p.name) for p in tparams]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(tparams, updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, inp, cost, extra):
    gshared = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad'%p.name) for p in tparams]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, [cost]+extra, updates=gsup, profile=False)

    pup = [(p, p - lr * g) for p, g in zip(tparams, gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=False)

    return f_grad_shared, f_update


### Model
class RNN(object):
    def __init__(self, options):
        self.options=options

    def init_parameters(self):
        if hasattr(self, "parameters"):
            return self.parameters

        ''' Create parameters as numpy variables'''
        h0_val = zeros((self.options.num_hidden,))
        Wh_val = orthogonal((self.options.num_hidden, args.num_hidden))
        Wx_val = orthogonal((1, self.options.num_hidden))
        b_val = zeros((self.options.num_hidden,))

        ''' Create parameters as numpy variables'''
        self.h0 = theano.shared(h0_val, name="h0")
        self.Wh = theano.shared(Wh_val, name="Wh")
        self.Wx = theano.shared(Wx_val, name="Wx")
        self.b = theano.shared(b_val, name="b")


        ''' Create parameters list '''
        self.parameters = [self.h0, self.Wh, self.Wx, self.b]
        return self.parameters


    def fprop(self, x):

        ''' Assume x is in the (t,b,c) order  '''

        ''' Recursive Functions '''
        ''' Arguments order should be: sequences, outputs,  nonsequences  '''
        def stepfn(xtilde, h, Wh, b):
            htilde = T.dot(h, Wh)
            h =  T.tanh(xtilde + htilde + b)
            return h

        ''' Project the input using self.Wx '''
        xtilde = T.dot(x, self.Wx)

        ''' Symbolic Loop using scan '''
        h, _ = theano.scan(stepfn,
                             sequences=[xtilde],
                             non_sequences=[self.Wh, self.b],
                             outputs_info=[T.repeat(self.h0[None, :], xtilde.shape[1], axis=0)],
                             strict=True)
        return h

class LSTM(object):
    def __init__(self, options):
        self.options=options

    def init_parameters(self):
        if hasattr(self, "parameters"):
            return self.parameters

        self.h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
        self.c0 = theano.shared(zeros((args.num_hidden,)), name="c0")
        self.Wh = theano.shared(orthogonal((args.num_hidden, 4 * args.num_hidden)), name="Wa")
        self.Wx = theano.shared(orthogonal((1, 4 * args.num_hidden)), name="Wx")
        self.ab_betas = theano.shared(zeros((4 * args.num_hidden,)), name="ab_betas")

        # forget gate bias initialization
        forget_biais = self.ab_betas.get_value()
        forget_biais[self.options.num_hidden:2*self.options.num_hidden] = 2.1
        self.ab_betas.set_value(forget_biais)


        ''' Create parameters list '''
        self.parameters = [self.h0, self.c0, self.Wh, self.Wx, self.ab_betas]
        return self.parameters


    def fprop(self, x):

        ''' Assume x is in the (t,b,c) order  '''

        ''' Recursive Functions '''
        ''' Arguments order should be: sequences, outputs,  nonsequences  '''

        def stepfn(xtilde, h, c, Wh, ab_betas):
            ab = T.dot(h, Wh) + xtilde + ab_betas

            g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * self.options.num_hidden])
                          for j, fn in enumerate([T.tanh] + 3 * [T.nnet.sigmoid])]
            c = f * c + i * g
            h = o * T.tanh(c)
            return h, c

        ''' Project the input using self.Wx '''
        xtilde = T.dot(x, self.Wx)

        ''' Symbolic Loop using scan '''
        [h, c],  _ = theano.scan(stepfn,
                                 sequences=[xtilde],
                                 non_sequences=[self.Wh, self.ab_betas],
                                 outputs_info=[T.repeat(self.h0[None, :], xtilde.shape[1], axis=0),
                                               T.repeat(self.c0[None, :], xtilde.shape[1], axis=0)],
                                 strict=True)
        return h



class Model(object):
    def __init__(self, options):
        self.options = options

        if self.options.lstm:
            self.rnn = LSTM(options=self.options)
        else:
            self.rnn = RNN(options=self.options)


    def init_parameters(self):

        ''' Initialize model param list '''
        self.parameters = []

        ''' RNN Layer '''
        self.rnn.init_parameters()
        self.parameters += self.rnn.parameters

        ''' FC Layer '''
        nclasses = 10
        self.Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy_c")
        self.by = theano.shared(zeros((nclasses,)), name="by_c")
        self.parameters += [self.Wy, self.by]

        ''' Return parameters list '''
        return self.parameters




    def fprop(self, x, y):
        length = 784
        ''' Resahape x into a sequence of 784 pixel (b, 784, 1) '''
        x = x.reshape((x.shape[0], length, 1))
        ''' Put the time dimension first '''
        x = x.dimshuffle(1, 0, 2)
        ''' Flatten the channel '''
        y = y.flatten(ndim=1)
        '''' Call the fprop of rnn '''
        h = self.rnn.fprop(x)

        if self.options.permuted:
            permutation = np.random.randint(0, length, size=(length,))
            x = x[permutation]

        '''' Compute ytilde from the last state h '''
        ytilde = T.dot(h[-1], self.Wy) + self.by
        ''' Applied softmax on ytilde'''
        yhat = T.nnet.softmax(ytilde)

        ''' Cost functions '''
        errors = T.neq(y, T.argmax(yhat, axis=1))
        cross_entropies = T.nnet.categorical_crossentropy(yhat, y)


        return cross_entropies.mean(), errors.mean()




    def build(self):
        if self.options.debug:
            theano.config.compute_test_value = "warn"

        ''' Create parameters '''
        self.init_parameters()

        ''' Create Theano Variables '''
        x = T.tensor4("features")
        y = T.imatrix("targets")

        batch = next(get_stream(which_set="train", batch_size=args.batch_size).get_epoch_iterator())
        x.tag.test_value = batch[0]
        y.tag.test_value = batch[1]


        ''' Construct theano grah '''
        [nll, err] = self.fprop(x, y)

        '''Compute gradient'''
        grads = T.grad(nll, wrt=self.parameters)
        grads_inputs = T.grad(nll, wrt=x)
        grads_inputs_r = grads_inputs.reshape((grads_inputs.shape[0], 784))

        ''' Gradient Clipping'''
        if self.options.clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(
                    T.switch(g2 > (self.options.clip_c**2),
                             g / T.sqrt(g2) * self.options.clip_c, g))
            grads = new_grads


        ''' Compile theano functions '''
        self.f_grad_shared, self.f_update = eval(self.options.optimizer)(
            T.scalar(name='lr'), self.parameters, grads, [x, y], nll, extra=[err])
        self.f_eval = theano.function([x, y], [nll, err], name='f_eval', on_unused_input='warn')
        self.f_grad_inputs = theano.function([x, y], grads_inputs_r.norm(2, axis=0))


class MainLoop(object):
    def __init__(self, options):
        self.options = options

        ## Get datasets informations through Fuel
        self.train_stream =get_stream(which_set="train",
                                      batch_size=options.batch_size)
        self.valid_stream =get_stream(which_set="valid",
                                      batch_size=options.batch_size)

        ## Create Model Class
        self.model = Model(self.options)
        ## Create model shared parameters and compiles model functions
        self.model.build()



    def validation(self):
        nlls = []
        errs = []

        # Iterate over the validation examples
        data_it = self.valid_stream.get_epoch_iterator()
        for counter, data in enumerate(data_it):
            ### Evaluate the model on the data
            nll, err = self.model.f_eval(*data)
            nlls += [float(nll)]
            errs += [float(err)]
            print 'validation %d'% (counter)
        return np.mean(nlls), np.mean(errs)


    def run(self):

        max_epoch = 100

        for epoch_counter in xrange(max_epoch):

            train_it =  self.train_stream.get_epoch_iterator()
            update_counter = 0
            moving_cost = 0


            for data in train_it:

                ### For dbg
                if False:
                    f1 = plt.figure()
                    ax1 = f1.add_subplot(111)
                    plt.title(" Gradient")
                    plt.ylabel('Norm')
                    plt.xlabel(" Iteration")
                    grads_norm = self.model.f_grad_inputs(*data)
                    ax1.plot(np.arange(784), grads_norm, linewidth=3)
                    plt.show()
                    exit(1)

                [nll, err ] = self.model.f_grad_shared(*data)
                self.model.f_update(self.options.learning_rate)

                if np.isnan(nll):
                    print 'NaN cost'
                    import pdb; pdb.set_trace()
                if update_counter == 0:
                    moving_cost = np.mean(nll)
                    moving_err = np.mean(err)
                else:
                    moving_cost = moving_cost * 0.95 + np.mean(nll) * 0.05
                    moving_err = moving_err  * 0.95 + np.mean(err) * 0.05

                print '(epoch %d, %d, moving cost %.3f, moving err %.3f' % (epoch_counter, update_counter,
                                                                            moving_cost, moving_err)
                update_counter += 1

            nll, err = self.validation()
            print '(epoch %d, validation cost %.3f, validation err %.3f' % (epoch_counter, nll, err)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-hidden", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--permuted", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    args = parser.parse_args()

    args.optimizer = 'rmsprop'
    args.clip_c = 10.

    np.random.seed(args.seed)
    main_loop = MainLoop(args)
    main_loop.run()

