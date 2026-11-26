---
license: apache-2.0
task_categories:
- text-classification
language:
- en
size_categories:
- 10K<n<100K
---

## LIAR2

The [LIAR](https://doi.org/10.18653/v1/P17-2067) dataset has been widely followed by fake news detection researchers since its release, and along with a great deal of research, the community has provided a variety of feedback on the dataset to improve it. We adopted these feedbacks and released the LIAR2 dataset, a new benchmark dataset of ~23k manually labeled by professional fact-checkers for fake news detection tasks. We have used a split ratio of 8:1:1 to distinguish between the training set, the test set, and the validation set, details of which are provided in the paper of "[An Enhanced Fake News Detection System With Fuzzy Deep Learning](https://doi.org/10.1109/ACCESS.2024.3418340)". The LIAR2 dataset can be accessed at [Huggingface](https://huggingface.co/datasets/chengxuphd/liar2) and [Github](https://github.com/chengxuphd/LIAR2),

## Example Usage

You can load each of the subset as follows:

```python
import datasets

dataset = "chengxuphd/liar2"
dataset = datasets.load_dataset(dataset)

statement_train, y_train = dataset["train"]["statement"], dataset["train"]["label"]
statement_val, y_val = dataset["validation"]["statement"], dataset["validation"]["label"]
statement_test, y_test = dataset["test"]["statement"], dataset["test"]["label"]
```

## Citation

If you find our work useful in your research, please consider citing:


```bibtex
@article{xu2024enhanced,
  author={Xu, Cheng and Kechadi, M-Tahar},
  journal={IEEE Access}, 
  title={An Enhanced Fake News Detection System With Fuzzy Deep Learning}, 
  year={2024},
  volume={12},
  number={},
  pages={88006-88021},
  keywords={Fake news;Fuzzy logic;Benchmark testing;Social networking (online);Deep learning;Task analysis;Natural language processing;Classification algorithms;Deep learning;fuzzy deep learning;fake news;fake news detection;fact-checking;NLP;classification systems;benchmark},
  url={https://doi.org/10.1109/ACCESS.2024.3418340},
  doi={10.1109/ACCESS.2024.3418340}}
```


```bibtex
@inproceedings{xu2023fuzzy,
   author = {Xu, Cheng and Kechadi, M-Tahar},
   title = {Fuzzy Deep Hybrid Network for Fake News Detection},
   year = {2023},
   isbn = {9798400708916},
   publisher = {Association for Computing Machinery},
   address = {New York, NY, USA},
   url = {https://doi.org/10.1145/3628797.3628971},
   doi = {10.1145/3628797.3628971},
   booktitle = {Proceedings of the 12th International Symposium on Information and Communication Technology},
   pages = {118–125},
   numpages = {8},
   keywords = {Classification Systems, Deep Learning, Hybrid Learning Models, Fuzzy Deep Learning, Fake News Detection},
   location = {<conf-loc>, <city>Ho Chi Minh</city>, <country>Vietnam</country>, </conf-loc>},
   series = {SOICT '23}
}
```