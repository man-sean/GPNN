import gif
import shlex
import argparse
import numpy as np

from pprint import PrettyPrinter
from matplotlib import pyplot as plt

from model.gpnn import gpnn
from model.parser import *

gif.options.matplotlib["dpi"] = 75


def imshow(img, title=None, show=True):
    @gif.frame
    def imshow_(img, title, show):
        img = img.copy()
        if img.dtype in (np.float64, np.float32):
            if img.min() >= 0 and img.max() <= 1:
                img *= 255
            img = np.uint8(np.clip(img, 0, 255))
        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.tight_layout()
        if show:
            plt.show()
    frame = imshow_(img, title, show=False)
    if show:
        imshow_(img, title, show=True)
    return frame


if __name__ == '__main__':
    cmd = shlex.split(' '.join([
        '-in database/tree.png',
        # '-in database/cows.png',
        '--out_size 125',
        '--patch_size 7',
        '--alpha 0.9',
        '--beta 1.5',
        # '--faiss',
        # '--ref-img sketches/tree.png',
        '--ref-img sketches/singan_tree.png',
    ]))
    parser = argparse.ArgumentParser()
    parser = parser_general(parser)
    # parser = parser_sample(parser)
    parser = parser_sketch(parser)
    config = vars(parser.parse_args(cmd))

    pp = PrettyPrinter()
    print('Config:')
    pp.pprint(config)

    model = gpnn(config)
    _ = model.run(to_save=True)
    frames = [imshow(np.concatenate([model.x_pyramid[-1], model.coarse_img.clip(0, 1)], axis=1),
                     title=f'Coarse Image (clipped)')]
    levels = [(idx, x, y) for idx, (x, y) in enumerate(zip(model.x_pyramid, model.y_pyramid))]
    for idx, x, y in reversed(levels):
        frames.append(imshow(np.concatenate([x, y], axis=1), title=f'Level {idx}'))
    gif.save(frames, f"{model.out_file.rsplit('.', maxsplit=1)[0]}.gif",
             duration=len(frames), unit="s", between="startend")

    frames = [imshow(model.coarse_img.clip(0, 1), title=f'Coarse Image (clipped)')]
    for level in reversed(range(len(model.intermidate))):
        for idx,  iter in enumerate(model.intermidate[level]):
            frames.append(imshow(iter, title=f'Level {level} Iterations {idx}', show=False))
    gif.save(frames, f"{model.out_file.rsplit('.', maxsplit=1)[0]}_full.gif",
             duration=len(model.intermidate)+1, unit="s", between="startend")
