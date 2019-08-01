CFLAGS='-stdlib=libc++' python setup.py build
# ^ --force to rebuild each time
python setup.py install
python src/run.py