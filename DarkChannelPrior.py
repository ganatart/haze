import cv2
import numpy as np

from PIL import Image

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
		>>> img = Image.open("example.jpg")
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
		>>> img = Image.open("example.jpg")
		>>> AtmLight(img)
		(62, 62, 71)

		>>> img = Image.open("example.jpg")
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

def main():
	TEST_IMAGE_PATH = "./example.jpg"

	# extract dark channel
	img = Image.open(TEST_IMAGE_PATH)
	dcp = DarkChannel(img, shape = "rect")
	dcp.save("dark_channel.png")

	# extract atm light
	img = np.array(img).astype(np.float32) / 255
	atm = AtmLight(img)
	print("Atm Light", atm)

if __name__ == "__main__":
	main()