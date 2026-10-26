# 1. Get data
## 1.1 download data
For this tutorial we will download open Metäkeskus data for the Pirkanmaa maakuknta, as instructed [here]("how_to_generate_allometry_from_metsakeskus.md").
[Here's the direct link for the download (1.1 GB)](https://avoin.metsakeskus.fi/aineistot/MV/Maakunta/MV_Pirkanmaa.zip).

TODO: Perhaps in the future we can have a dataset project (https://docs.csc.fi/computing/roihu-dataset-project/#applying-for-a-dataset-project) and skip this tedious manual step.

## 1.2 upload data to CSC
There are many ways how to move data to CSC, all documented in [this list](https://docs.csc.fi/data/moving/).
We are going to use [scp](https://docs.csc.fi/data/moving/scp/).

Before we pass any data, we must establish a connection to Roihu by obtaining a SSH certificate which is valid for 24 hours ([read the docs](https://docs.csc.fi/computing/connecting/ssh-keys/#signing-public-key)).


