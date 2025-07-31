"""Main"""

# imports: library
from argparse import ArgumentParser
from .testing import run_tests

from . import version
import os
from . detect import batch_extract_embeddings
import numpy as np


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
	parser.add_argument("--output-directory", help = "Directory to save embedding databases (defaults to current directory", default = None)

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
		
		out_dir = args.output_directory
		if out_dir is None:
			out_dir = os.getcwd()
		
		for target_model in [args.model]:
			if args.generate_embeddings:
				assert os.path.isdir(args.path)
				absolute_files = []
				for root, dirs, files in os.walk(args.path):
					
					for fn in files:
						absolute_files.append(os.path.join(root, fn) )
				file_refs, embeddings = batch_extract_embeddings(absolute_files, args.batch_size, target_model)
				db_filename = os.path.join(out_dir, f"db_{target_model}_{len(file_refs)}.npz")
				np.savez_embeddings(db_filename, file_refs = file_refs, embeddings = embeddings)
				print(f"Wrote embedding database for {target_model} to `{db_filename}`")
			else:
				# verify images
				raise NotImplementedError

if __name__ == '__main__':
	main()
