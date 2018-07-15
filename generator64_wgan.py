import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from utils import Gamma_initializer


class Generator(chainer.Chain):
    """Generator

    build Generator model

    Parametors
    ---------------------
    n_hidden: int
       dims of random vector z

    bottom_width: int
       Width when converting the output of the first layer
       to the 4-dimensional tensor

    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    Attributes
    ---------------------
    """

    def __init__(self, n_hidden=100, bottom_width=4, ch=512, ksize=4, pad=1):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            # initializers
            transposed_conv_init = chainer.initializers.Normal(scale=0.02)
            gamma_init = Gamma_initializer(mean=1.0, scale=0.02)
            beta_init = chainer.initializers.Zero()

            self.dc0 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch,
                ksize=4,
                pad=0,
                stride=1,
                initialW=transposed_conv_init,
                nobias=True)  # (, 512, 4, 4)
            self.dc1 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 2,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=transposed_conv_init,
                nobias=True)  # (, 256, 8, 8)
            self.dc2 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 4,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=transposed_conv_init,
                nobias=True)  # (, 128, 16, 16)
            self.dc3 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 8,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=transposed_conv_init,
                nobias=True)  # (, 64, 32, 32)
            self.dc4 = L.Deconvolution2D(
                in_channels=None,
                out_channels=3,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=transposed_conv_init,
                nobias=True)  # (, 3, 64, 64)

            self.bn0 = L.BatchNormalization(
                ch,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05)
            self.bn1 = L.BatchNormalization(
                ch // 2,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05)
            self.bn2 = L.BatchNormalization(
                ch // 4,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05)
            self.bn3 = L.BatchNormalization(
                ch // 8,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05)

    def make_hidden(self, batchsize):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        batchsize: int
           batchsize indicate len(z)

        """
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
                        .astype(np.float32)

    def __call__(self, z):
        """
        Function that computs foward

        Parametors
        ----------------
        z: Variable
           random vector drown from a uniform distribution,
           this shape is (N, 100)

        """
        h = F.relu(self.bn0(self.dc0(z)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.tanh(self.dc4(h))
        return x


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable

    model = Generator()
    img = model(Variable(model.make_hidden(10)))
    # print(img)
    g = c.build_computational_graph(img)
    with open('gen_graph.dot', 'w') as o:
        o.write(g.dump())
