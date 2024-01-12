import cv2
import torch
import numpy as np
from skimage.color import rgb2gray

from PIL import Image

try:
	from zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
except:
	raise ImportError("Error while importing zoedepth, try download zoedepth using shell script.")

DEVICE = "cuda" if torch.cuda.is_available else "cpu"

def DarkChannel(img: Image.Image | np.ndarray, shape: str = "rect", ksize: int = 15) -> Image.Image | np.ndarray:
	"""
		Extract Dark Channl Prior from image

		Parameters
		----------
		img: Image.Image | np.ndarray
			image to extrack DCP (H, W, 3)
		shape: str
			kernel shape for DCP, should be one of "rect" or "ellipse"
		ksize: int
			kernel size for DCP

		Returns
		-------
		dcp: Image.Image | np.ndarray
			dark channel prior of given image

		Examples
		--------
		>>> img = Image.open("haze.jpg")
		>>> dcp = DarkChannelPrior(img)
		>>> dcp.save("dark_channel.png")
	"""

	is_numpy = type(img) == np.ndarray

	if not is_numpy:
		# convert image to numpy array
		img = img.convert("RGB")
		img = np.array(img)
	
	# get min for every channel
	channel_min = np.min(img, axis = 2)

	# get min for every patch
	match shape:
		case "rect":
			shape = cv2.MORPH_RECT
		case "ellipse":
			shape = cv2.MORPH_ELLIPSE
		case _:
			raise ValueError("shape should be one of rect or ellipse, but given shape is {}.".format(shape))
	kernel = cv2.getStructuringElement(shape, (ksize, ksize))
	dark_channel = cv2.erode(channel_min, kernel)

	if not is_numpy:
		# convert image to PIL image
		dark_channel = Image.fromarray(dark_channel, mode = "L")
	
	return dark_channel

def AtmLight(img: Image.Image | np.ndarray, dark_channel: Image.Image | None = None, ratio: float = 0.001) -> tuple[int, int, int] | tuple[float, float, float]:
	"""
		Estimate color of Atm image using dark channel
		
		Parameters
		----------
		img: Image.Image | np.ndarray:
			Image to estimate atm light
		dark_channel: Image.Image | None
			Dark channel to estimate atm light
			If not given dark channel, then it also calculate dark channel from given image
		ratio: float
			Ratio of pixels from image used to estiamate atm light
		
		Returns
		-------
		atm: tuple[int, int, int] | tuple[float, float, float]
			estimated atm light of given image
			If input image is PIL image or int numpy array, then return int tuple
			If input image is float32 numpy array, then return float tuple

		Examples
		--------
		>>> img = Image.open("haze.jpg")
		>>> AtmLight(img)
		(62, 62, 71)

		>>> img = Image.open("haze.jpg")
		>>> img = np.array(img).astype(np.float32) / 255
		>>> AtmLight(img)
		(0.246818, 0.24618295, 0.27960142)

	"""
	# check parameters
	if ratio <= 0 or ratio > 1:
		raise ValueError("Ratio should be in range (0, 1], but given ratio is {}".format(ratio))

	is_numpy = type(img) == np.ndarray

	if not is_numpy:
		# convert image into numpy array
		img = img.convert("RGB")
		img = np.array(img)
	
	if dark_channel is None:
		# calculate dark channel if not given
		dark_channel = DarkChannel(img)
	else:
		# convert image into numpy array
		dark_channel.convert("L")
		dark_channel = np.array(dark_channel)

	is_float = img.dtype == np.float32

	if not is_float:
		# convert int image into float image
		img = img.astype(np.float32) / 255
	
	# extract atm using dark channel
	img_vec = img.reshape((-1, 3))
	dark_vec = dark_channel.reshape((-1))
	indices = dark_vec.argsort()[:int(len(dark_vec) * ratio)]
	atm = img_vec[indices]
	atm = np.mean(atm, axis = 0)

	if not is_float:
		# convert atm to int
		atm = (atm * 255).astype(np.uint8)
	
	return tuple(atm)

def GuidedFilter(guide: Image.Image | np.ndarray, img: Image.Image | np.ndarray, sz: int = 15, eps: float = 1e-6) -> Image.Image | np.ndarray:
	"""
		Perform guided filtering for given image and guidance

		Parameters
		----------
		guide: Image.Image | np.ndarray
			Image to be used as an guidance
		img: Image.Image | np.ndarray
			Image to perform filtering
		ksize: int
			Size of filter kernel
		eps: float
			Small value of float to prevent zero division error
		
		Returns
		-------
		res: Image.Image | np.ndarray
			Result of guided filtering
			return Image if given img is Image
			return float32 type numpy array if given img is numpy

		Examples
		--------
		>>> img = Image.open("img.png")
		>>> dcp = Image.open("dcp.png")
		>>> filtered = GuidedFilter(img, dcp)
	"""

	is_numpy = type(img) == np.ndarray

	I = guide
	p = img

	# convert guide image into gray float scale image
	
	I = np.array(guide)
	if I.dtype == np.uint8:
		I = I.astype(np.float32) / 255
	I = rgb2gray(I)

	# convert input image into float scale image
	p = np.array(p)
	if p.dtype == np.uint8:
		p = p.astype(np.float32) / 255

	# perform guided filtering
	meanI = cv2.blur(I, (sz, sz))
	meanP = cv2.blur(p, (sz, sz))
	corrI = cv2.blur(I * I, (sz, sz))
	corrIp = cv2.blur(I * p, (sz, sz))

	varI = corrI - meanI * meanI
	covIp = corrIp - meanI * meanP

	a = covIp / (varI + eps)
	b = meanP - a * meanI

	meanA = cv2.blur(a, (sz, sz))
	meanB = cv2.blur(b, (sz, sz))

	res = meanA * I + meanB
	res = np.clip(res, 0, 1)

	if not is_numpy:
		# convert numpy image into Image.Image
		res = (res * 255).astype(np.uint8)
		res = Image.fromarray(res)	

	return res

def GetDepth(img: Image.Image | np.ndarray, model: ZoeDepth, filter: bool = True):
	"""
		Perform guided filtering for given image and guidance

		Parameters
		----------
		guide: Image.Image | np.ndarray
			Image to be used as an guidance
		img: Image.Image | np.ndarray
			Image to perform filtering
		ksize: int
			Size of filter kernel
		eps: float
			Small value of float to prevent zero division error
		
		Returns
		-------
		res: Image.Image | np.ndarray
			Result of guided filtering
			return Image if given img is Image
			return float32 type numpy array if given img is numpy

		Examples
		--------
		>>> img = Image.open("img.png")
		>>> dcp = Image.open("dcp.png")
		>>> filtered = GuidedFilter(img, dcp)
	"""
	
	if DEVICE == "cpu":
		raise UserWarning("CUDA device is not available. Loading model on cpu.")

	depth = model.infer_pil(img)
	depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

	depth = (depth * 255).astype(np.uint8)
	depth = Image.fromarray(depth)
	depth.save("depth.png")
	
def main():
	CLEAR_IMAGE_PATH = "./clear.png"
	HAZE_IMAGE_PATH = "./haze.png"

	# extract dark channel
	clear = Image.open(CLEAR_IMAGE_PATH)
	model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained = True, verbose = False)

	GetDepth(clear, model_zoe_n.to(DEVICE))

if __name__ == "__main__":
	main()