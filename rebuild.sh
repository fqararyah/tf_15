python3 -m pip uninstall numpy -y
python3 -m pip install -U --user pip numpy==1.19.5 wheel
python3 -m pip install -U --user keras_preprocessing --no-deps
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /home/nahmad/tensorflow_pkg3/15
python3 -m pip uninstall tensorflow -y
python3 -m pip install /home/nahmad/tensorflow_pkg3/15/tensorflow-1.15.3-cp37-cp37m-linux_x86_64.whl