TARGET = leaffliction.keras

# Usage:
# make ARGS=some_image.JPG

# Or without make
# uv sync
# source .venv/bin/activate
# python file/Train.py file/images
# python file/Predict.py someimage.JPG

.PHONY: infer
infer: train
	. .venv/bin/activate; python file/Predict.py $(ARGS)

.PHONY: train
train : $(TARGET)

$(TARGET): images
	uv sync
	. .venv/bin/activate; python file/Train.py file/images

.PHONY: images
images :
	wget -O file/leaves.zip https://cdn.intra.42.fr/document/document/39824/leaves.zip
	unzip file/leaves.zip -d file
	rm -r file/leaves.zip
