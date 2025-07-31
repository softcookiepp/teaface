"""Main"""

# imports: library
from argparse import ArgumentParser
from .testing import run_tests

from . import version
import os
from . detect import batch_extract_embeddings


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
	parser.add_argument("--model", help = "Model to use for embedding generation or querying", default = None)
	parser.add_argument("--db", help = "Embedding database to use")
	parser.add_argument("--batch-size", help = "Batch size to use for embedding generation", default = 64, type = int)

	args = parser.parse_args()

	if args.version:
		print(f'{version.PROGRAM_NAME} {version.__version__}')
		return
	if args.test:
		run_tests()
	else:
		assert not args.path is None, "Path must be specificed using the `--path` argument."
		if args.model is None:
			args.model = "ArcFace"
		assert args.model in ["ArcFace"]
		if args.generate_embeddings:
			assert os.path.isdir(args.path)
			absolute_files = []
			for root, dirs, files in os.walk(args.path):
				
				for fn in files:
					absolute_files.append(os.path.join(root, fn) )
			file_refs, embeddings = batch_extract_embeddings(absolute_files, args.batch_size, args.model)
			input(embeddings)
		else:
			# verify images
			raise NotImplementedError

if __name__ == '__main__':
	main()
