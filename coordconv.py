from __future__ import print_function
import torch
import numpy as np

class Coords():
	"""Add coords to a tensor"""
	def __init__(self, x_dim=64, y_dim=64, with_r=False):
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.with_r = with_r

	def call(self, input_tensor):
		batch_size, _, x_dim, y_dim = input_tensor.size()

		xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
		yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

		xx_channel = xx_channel.float() / (x_dim - 1)
		yy_channel = yy_channel.float() / (y_dim - 1)

		xx_channel = xx_channel * 2 - 1
		yy_channel = yy_channel * 2 - 1

		xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
		yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

		ret = torch.cat([
		    input_tensor,
		    xx_channel.type_as(input_tensor),
		    yy_channel.type_as(input_tensor)], dim=1)

		if self.with_r:
			rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
			ret = torch.cat([ret, rr], dim=1)
		return ret
