import chainer
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    """
    Discriminator applying GAP and nobias to discriminator.Discrimiator
    """

    def __init__(self, bottom_width=128, ch=1024, wscale=0.02):
        super(Discriminator, self).__init__()
        print("Discriminator applying GAP to output layer")
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
                nobias=True,
                initialW=w)  # (, 1, 4, 4)
            self.c5 = L.Convolution2D(
                in_channels=None,
                out_channels=1,
                ksize=1,
                stride=1,
                pad=0,
                initialW=w)  # (, 1, 4, 4)

            self.bn1 = L.BatchNormalization(size=ch // 8)
            self.bn2 = L.BatchNormalization(size=ch // 4)
            self.bn3 = L.BatchNormalization(size=ch // 2)
            self.bn4 = L.BatchNormalization(size=ch)

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
        h = F.leaky_relu(self.bn4(self.c4(h)))
        # mimcing global max pooling with average_pooling_2d
        h = self.c5(h)
        logits = F.average_pooling_2d(h, ksize=(4, 4))
        logits = F.squeeze(logits, axis=(1, 2, 3))  # scalarにするため.必要なし?

        return logits


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    # batch データが1つでtrain:Trueの時にBNに通すとWarningが出る
    # https://github.com/chainer/chainer/pull/3996 のこと
    z = np.random.uniform(-1, 1, (1, 3, 128, 128)).astype("f")
    labels = np.array([1])
    model = Discriminator()
    img = model(Variable(z))

    # print(img)
    g = c.build_computational_graph(img)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
