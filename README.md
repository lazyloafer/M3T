## Multi-hop Multi-view Memory Transformer for Session-based Recommendation

codes and datasets for AAAI-2023 anonymous reviewing

## Dependencies

- Python (>=3.7)

- PyTorch (>=1.9)

## Datasets
You can download the datasets from:
* Gowalla: http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
* LastFM: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
* Yoochoose: https://www.kaggle.com/chadgostopp/recsys-challenge-2015
* You can process Gowalla and LastFM referring to https://github.com/SpaceLearner/SessionRec-pytorch
* You can process Yoochoose1/64 referring to https://github.com/CRIPAC-DIG/SR-GNN

## Parameter setting
* Gowalla: --order_list=[1,2,3]
* LastFM: --order_list=[1,2,3,4,5]
* Yoochoose1/64: --order_list=[1,2]

## Process
 - Place the datasets in `datasets`
 - Train a model:
 ```bash
 python main.py
```

