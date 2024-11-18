# DNA-MSAN

The official implementation for paper "Double Negative Sampled Graph Adversarial Representation Learning with Motif-based Structural Attention Network".

### Requirements

During the experiments, the runtime environment and core library versions we used were as follows:

```bibtex
matplotlib==3.1.1
numpy==1.15.1
sqlite==3.40.1
pillow==8.3.2
python==3.6.8
scikit-learn==0.22.1
scipy==1.3.2
torch==1.10.2
torchvision==0.2.2.post3
tornado==6.1
tqdm==4.63.0
```

### Dataset

You can download the Cora, Citeseer, and Pubmed datasets through the following links: **https://linqs.org/datasets/**

For the LastFM dataset, you can download it via the following link: **https://snap.stanford.edu/data/feather-lastfm-social.html**

For the Wiki dataset, as it does not contain feature attributes, we use an identity matrix as the feature matrix for the original graph.

### Citation

If you find our code and model useful, please cite our work. Thank you!
