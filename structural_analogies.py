import argparse
import shlex

from model.gpnn import gpnn
from model.parser import *
from skimage.transform import rescale, resize

from experimenting import imshow

if __name__ == '__main__':
	cmd = shlex.split(' '.join([
		'-b database/sand.jpg',
		'-a database/snow.jpg',
		# '-b sketches/tree.png',
		'--out_size 125',
	]))
	parser = argparse.ArgumentParser()
	parser = parser_general(parser)
	parser = parser_analogies(parser)
	config = vars(parser.parse_args(cmd))
	model = gpnn(config)
	refine_img = model.run(to_save=False)
	model.coarse_img = resize(refine_img, model.coarse_img.shape[:2])
	img = model.run()
