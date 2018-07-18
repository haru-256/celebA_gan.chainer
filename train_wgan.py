import chainer
from chainer import training
from chainer.training import extensions
from chainer.datasets import ImageDataset
from chainer.serializers import save_npz
from generator64_wgan import Generator
from critic64 import Critic
from updater import WGANUpdater
from utils import out_generated_image
# from accuracy_reporter import accuracy_report
import pathlib
import matplotlib.pyplot as plt
import pathlib
plt.style.use("ggplot")


def make_optimizer(model):
    """
    Setup an optimizer
    """
    optimizer = chainer.optimizers.RMSprop(lr=0.00005, alpha=0.99, eps=1e-8)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')

    return optimizer


if __name__ == '__main__':
    import numpy as np
    import argparse

    # パーサーを作る
    parser = argparse.ArgumentParser(
        prog='train',  # プログラム名
        usage='train WGAN',  # プログラムの利用方法
        description='description',  # 引数のヘルプの前に表示
        epilog='end',  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # 引数の追加
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('--hidden', help='the number of codes of Generator.',
                        type=int, default=100)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 25',
                        type=int, default=25)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 64',
                        type=int, default=64)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0',
                        choices=[0, 1], type=int, default=0)
    parser.add_argument('-ks', '--ksize',
                        help='specify kernel size of generator by this number. any of following;'
                        '4 or 6. d defalut value is 4',
                        choices=[4, 6], type=int, default=4)
    parser.add_argument('-nc', '--n_critic',
                        help='specify number of iteretion of critic by this number.'
                        ' defalut value is 5',
                        type=int, default=5)
    parser.add_argument('-c_l', '--clip_lower',
                        help='specify lower of clip range by this number.',
                        type=float, default=-0.01)
    parser.add_argument('-c_u', '--clip_upper',
                        help='specify upper of clip range by this number.',
                        type=float, default=0.01)
    parser.add_argument('-nb', '--nobias',
                        help='whether do\'t apply bias to convolution layer with no bias.',
                        action='store_false')
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)

    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    n_hidden = args.hidden
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    if args.ksize == 6:
        pad = 2
    else:
        pad = 1

    out = pathlib.Path("result_{0}".format(number))
    if not out.exists():
        out.mkdir()
    out /= pathlib.Path("result_{0}_{1}".format(number, seed))
    if not out.exists():
        out.mkdir()

    # 引数(ハイパーパラメータの設定)の書き出し
    with open(out / "args.txt", "w") as f:
        f.write(str(args))

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))
    print('# seed: {}'.format(seed))
    print('# ksize: {}'.format(args.ksize))
    print('# pad: {}'.format(pad))
    print('# nobias: {}'.format(args.nobias))

    # fix seed
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

    print('')

    # Set up a generator, critic
    gen = Generator(n_hidden=n_hidden, ksize=args.ksize,
                    pad=pad, nobias=args.nobias)
    critic = Critic(ksize=args.ksize, pad=pad, nobias=args.nobias)

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        critic.to_gpu()

    opt_gen = make_optimizer(gen)
    opt_critic = make_optimizer(critic)

    # Prepare Dataset
    data_dir = pathlib.Path("../data/CelebA/Img/center_cropped_resize_64/")
    abs_data_dir = data_dir.resolve()
    print("# data dir path:", abs_data_dir)
    data_path = [path for path in abs_data_dir.glob("*.jpg")]
    print("# data length:", len(data_path))
    data = ImageDataset(paths=data_path)  # dtype=np.float32

    # Prepare Iterator
    train_iter = chainer.iterators.SerialIterator(data, batch_size)

    # Set up a updater and trainer
    updater = WGANUpdater(
        models=(gen, critic),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen,
            'critic': opt_critic
        },
        device=gpu,
        n_critic=args.n_critic,
        clip_range=[args.clip_lower, args.clip_upper])
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    snapshot_interval = (10, 'epoch')
    display_interval = (1, 'epoch')
    # storage method is hdf5
    trainer.extend(
        extensions.snapshot(
            filename='snapshot_epoch_{.updater.epoch}.npz',
            savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(
            gen, 'gen_epoch_{.updater.epoch}.npz', savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(
            critic, 'critic_epoch_{.updater.epoch}.npz', savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'gen/loss', 'critic/loss', 'elapsed_time',
        ]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(
        out_generated_image(gen, 5, 5, seed, out),
        trigger=display_interval)
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss', 'critic/loss'],
            x_key='epoch',
            file_name='loss_{0}_{1}.jpg'.format(number, seed),
            grid=False))
    trainer.extend(extensions.dump_graph("gen/loss", out_name="gen.dot"))
    trainer.extend(extensions.dump_graph("critic/loss", out_name="critic.dot"))

    # Run the training
    trainer.run()
