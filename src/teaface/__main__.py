"""Main"""

# imports: library
from argparse import ArgumentParser
from .testing import run_tests

from . import version


def main() -> None:
	"""Main"""
	parser = ArgumentParser(prog=version.PROGRAM_NAME)

	parser.add_argument('--version',
						help='Display version',
						action='store_true',
						dest='version')
	parser.add_argument("--test", help="Run tests", action = "store_true", dest = "test")
	parser.add_argument("--generate-embeddings", help = "Generate an embedding database", action = "store_true")
	parser.add_argument("--path", help = "image/folder for querying or embedding generation", default = None)
	parser.add_argument("--model", help = "Model to use for embedding generation or querying", default = "all")

	args = parser.parse_args()

	if args.version:
		print(f'{version.PROGRAM_NAME} {version.__version__}')
		return
	if args.test:
		run_tests()
	else:
		assert not args.path is None, "Path must be specificed using the `--path` argument."
		if model is None:
			raise NotImplementedError
		raise NotImplementedError

if __name__ == '__main__':
	main()
