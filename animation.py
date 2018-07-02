import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import pathlib

fig = plt.figure(figsize=(10, 10))

path = pathlib.Path("result3_b/preview")

ims = []

for epoch in range(1, 101):
    img = cv2.imread(str(path / "image_{}epoch.png".format(epoch)))
    frame = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ims.append([frame])
ani = animation.ArtistAnimation(fig, ims, interval=250)
# ani.save('anim.mp4', writer="ffmpeg")
plt.axis("off")
ani.save('anim3_b.gif', writer="imagemagick")
plt.show()
