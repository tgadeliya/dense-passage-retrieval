
run_tests:
	pytest --ignore=./faiss

run_black:
	python -m black dpr

run_black_check:
	python -m black --check dpr

faiss/*:
	mkdir faiss
	git clone https://github.com/facebookresearch/faiss.git --single-branch ./faiss
	# If not presented add faiss source to .gitignore
	grep -qxF "#FAISS library" .gitignore || echo "\n#FAISS library\nfaiss/*" >> .gitignore

install_faiss_reqs:
	# Please refer to additional reqs on https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
	brew install openblas
	brew install swig
	brew install libomp

fix_swigfaiss_import:
	# TODO: Check whether it is necessary. Maybe newer libomp is the solution
	# Solution from https://github.com/facebookresearch/faiss/issues/866
	# Unzip egg-file
	unzip "$(pip show faiss|grep "Location"| cut -d " " -f2)"

build_faiss: install_faiss_reqs faiss/*
	echo "Building FAISS from sources"
	# Invoke CMake
	cmake -B faiss/build faiss/ -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON
	make -C ./faiss/build -j4 faiss
	# Python bindings
	make -C ./faiss/build -j4 swigfaiss
	(cd faiss/build/faiss/python && python setup.py install)


