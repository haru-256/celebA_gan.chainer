import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


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

    def __init__(self, n_hidden=100, bottom_width=4, ch=1024, wscale=0.02, ksize=6, pad=2):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)  # initializers
            # w = chainer.initializers.HeNormal()  # He initialize value
            self.l0 = L.Linear(
                None,
                self.ch * self.bottom_width * self.bottom_width,
                initialW=w,
                nobias=True)
            self.dc1 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 2,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=w,
                nobias=True)  # (, 512, 8, 8)
            self.dc2 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 4,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=w,
                nobias=True)  # (, 256, 16, 16)
            self.dc3 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 8,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=w,
                nobias=True)  # (, 128, 32, 32)
            self.dc4 = L.Deconvolution2D(
                in_channels=None,
                out_channels=ch // 16,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=w,
                nobias=True)  # (, 64, 64, 64)
            self.dc5 = L.Deconvolution2D(
                in_channels=None,
                out_channels=3,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=w)  # (, 3, 128, 128)
            self.bn0 = L.BatchNormalization(
                self.ch * self.bottom_width * self.bottom_width)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)
            self.bn4 = L.BatchNormalization(ch // 16)

    def make_hidden(self, batchsize):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        batchsize: int
           batchsize indicate len(z)

        """
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden))\
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
        h = F.relu(self.bn0(self.l0(z)))
        h = F.reshape(h, (len(z), self.ch, self.bottom_width,
                          self.bottom_width))  # dataformat is NCHW
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        x = F.tanh(self.dc5(h))
        return x


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable

    z = np.random.uniform(-1, 1, (1, 100)).astype("f")
    labels = Variable(np.array([2]))
    # labels = np.array([2])
    model = Generator(ksize=4, pad=1)
    img = model(Variable(z))
    # print(img)
    g = c.build_computational_graph(img)
    with open('gen_graph.dot', 'w') as o:
        o.write(g.dump())
