# GQA-dataloader
Data-preprocess for GQA
you can use the dataset in several steps

## 1.modify the config.py
you should set your own location as the input
## 2.merge the 16 object feature directory into 1
python merge.py --name object
## 3.test the data_gqa.py
directly test the data by using running data_gqa.py
