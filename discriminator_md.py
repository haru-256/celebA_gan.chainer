import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class Minibatch_Discrimination(chainer.Chain):
    """
    Minibatch Discrimination Layer

    Parameters
    ---------------------
    B: int
        number of rows of M

    C: int
        number of columns of M

    wscale: float
        std of normal initializer

    """

    def __init__(self, B, C, wscale):
        super(Minibatch_Discrimination, self).__init__()
        self.b = B
        self.c = C
        with self.init_scope():
            # initialozer to W
            w = chainer.initializers.Normal(wscale)

            # register Parameters
            self.t = L.Linear(in_size=None,
                              out_size=self.b*self.c,
                              initialW=w,
                              nobias=True)  # bias is required ?

    def __call__(self, x):
        """
        Calucurate Minibatch Discrimination using broardcast.

        Parameters
        ---------------
        x: Variable
           input vector shape is (N, num_units)
        """
        batch_size = x.shape[0]
        xp = x.xp
        x = F.reshape(x, (batch_size, -1))
        activation = F.reshape(self.t(x), (-1, self.b, self.c))

        m = F.reshape(activation, (-1, self.b, self.c))
        m = F.expand_dims(m, 3)
        m_T = F.transpose(m, (3, 1, 2, 0))
        m, m_T = F.broadcast(m, m_T)
        l1_norm = F.sum(F.absolute(m-m_T), axis=2)

        # eraser to erase l1 norm with themselves
        eraser = F.expand_dims(xp.eye(batch_size, dtype="f"), 1)
        eraser = F.broadcast_to(eraser, (batch_size, self.b, batch_size))

        o_X = F.sum(F.exp(-(l1_norm + 1e6 * eraser)), axis=2)

        # concatunate along channels or units
        return F.concat((x, o_X), axis=1)


class Discriminator(chainer.Chain):
    """
    Discriminator applied GAP and nobias, minibatch discrimination
     to discriminator.Discrimiator

    Parametors
    ---------------------
    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    wscale: float
        std of normal initializer

    B: int 
        number of rows of M

    C: int
        number of columns of M

    Attributes
    ---------------------

    Returns
    --------------------
    y: float
        logits

    """

    def __init__(self, bottom_width=128, ch=1024, wscale=0.02, B=32, C=8):
        super(Discriminator, self).__init__()
        self.b, self.c = B, C
        print(" Discriminator nobias, minibatch discrimination")
        with self.init_scope():
            # initializers
            w = chainer.initializers.Normal(wscale)

            # register layer with variable
            self.c0 = L.Convolution2D(
                in_channels=None,
                out_channels=ch // 16,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w)  # (, 64, 64, 64)
            self.c1 = L.Convolution2D(
                in_channels=None,
                out_channels=ch // 8,
                ksize=5,
                stride=2,
                pad=2,
                nobias=True,
                initialW=w)  # (, 128, 32, 32)
            self.c2 = L.Convolution2D(
                in_channels=None,
                out_channels=ch // 4,
                ksize=5,
                stride=2,
                pad=2,
                nobias=True,
                initialW=w)  # (, 256, 16, 16)
            self.c3 = L.Convolution2D(
                in_channels=None,
                out_channels=ch // 2,
                ksize=5,
                stride=2,
                pad=2,
                nobias=True,
                initialW=w)  # (, 512, 8, 8)
            self.c4 = L.Convolution2D(
                in_channels=None,
                out_channels=ch,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w)  # (, 1024, 4, 4)
            self.md5 = Minibatch_Discrimination(self.b, self.c, wscale)
            self.l6 = L.Linear(
                in_size=None,
                out_size=1,
                initialW=w
            )

            self.bn1 = L.BatchNormalization(size=ch // 8)
            self.bn2 = L.BatchNormalization(size=ch // 4)
            self.bn3 = L.BatchNormalization(size=ch // 2)
            # self.bn4 = L.BatchNormalization(size=ch)

    def __call__(self, x):
        """
        Function that computes forward

        Parametors
        ----------------
        x: Variable
           input image data. this shape is (N, C, H, W)
        """
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.leaky_relu(self.c4(h))
        h = self.md5(h)
        logits = self.l6(h)

        return logits


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    # batch データが1つでtrain:Trueの時にBNに通すとWarningが出る
    # https://github.com/chainer/chainer/pull/3996 のこと
    z = np.random.uniform(-1, 1, (10, 3, 128, 128)).astype("f")
    labels = np.array([1])
    model = Discriminator()
    img = model(Variable(z))

    # print(img)
    g = c.build_computational_graph(img)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
