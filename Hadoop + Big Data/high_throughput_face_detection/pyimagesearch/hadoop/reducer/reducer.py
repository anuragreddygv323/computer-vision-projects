class Reducer(object):
	@staticmethod
	def parse_mapper_output(stream, sep="\t"):
		# loop over the lines from the mapper output and yield each line
		for line in stream:
			yield line.rstrip().split(sep)