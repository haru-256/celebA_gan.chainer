import chainer
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    def __init__(self, bottom_width=128, ch=1024, wscale=0.02):
        super(Discriminator, self).__init__()
        print("Discriminator aplying Dense to outputlayer")
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
                initialW=w)  # (, 128, 64, 64)
            self.c2 = L.Convolution2D(
                in_channels=None,
                out_channels=ch // 4,
                ksize=5,
                stride=2,
                pad=2,
                nobias=True,
                initialW=w)  # (, 256, 64, 64)
            self.c3 = L.Convolution2D(
                in_channels=None,
                out_channels=ch // 2,
                ksize=5,
                stride=2,
                pad=2,
                nobias=True,
                initialW=w)  # (, 512, 64, 64)
            self.c4 = L.Convolution2D(
                in_channels=None,
                out_channels=ch,
                ksize=5,
                stride=2,
                pad=2,
                nobias=True,
                initialW=w)  # (, 1024, 64, 64)
            self.l5 = L.Linear(
                in_size=None, out_size=1, initialW=w)

            self.bn1 = L.BatchNormalization(size=ch // 8)
            self.bn2 = L.BatchNormalization(size=ch // 4)
            self.bn3 = L.BatchNormalization(size=ch // 2)
            self.bn4 = L.BatchNormalization(size=ch)
            # self.bn5 = L.BatchNormalization(size=1)  # case of aplying BN to outputlayer

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
        y = self.l5(h)  # conv->linear では勝手にreshapeが適用される
        # y = self.bn5(self.l5(h))  # aply BN to output layer

        return y


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
