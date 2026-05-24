<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<h1 align="center">Arctic River Rusting Detection using AlphaEarth Embeddings</h1>
<p align="center">
    Using Principal Component Analysis (PCA) on high-dimensional ML products.
  </p>
<br />
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#background">Background</a></li>
        <li><a href="#the-sentinel-2-satellite">The SENTINEL-2 Satellite</a></li>
        <li><a href="#machine-learning-methodology-k-means-clustering">Machine Learning Methodology: K-Means Clustering</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#datasets-used">Datasets Used</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
      <ul>
        <li><a href="#references">References</a></li>
      </ul>
  </ol>
</details>


# About the Project
This project is a final assignment for GEOL0069 at University College London, created to explore the usage of machine learning (ML) algorithms in Earth Sciences applications. It presents a scalable framework for detecting and characterising river rusting across Arctic environments using Google's Alpha Earth machine learning embeddings. The unsupervised learning method used to intepret the data is Principal Component Analysis (PCA).

## Background
The approach is inspired by the findings of O'Donell et al. (2024), [*"Metal mobilization from thawing permafrost to aquatic ecosystems is driving rusting of Arctic streams"*](https://www.nature.com/articles/s43247-024-01446-z#auth-Jonathan_A_-O_Donnell-Aff1), which identified widespread river colour changes linked to permafrost thaw in Alaska. Acid rock drainage conditions leading to iron and toxic heavy metal mobilisation are observed as previously frozen soil is exposed to an oxidizing environment. Iron-rich groundwater released from thawing permafrost alters river optical properties, producing strong orange-brown discolouration detectable from satellite imagery. 

<figure>
  <img width="1735" height="378" alt="Satellite imagery showing river rusting" src="https://github.com/user-attachments/assets/a124c0f9-1dbf-4ae2-8c6e-40712118ff16" />
  <figcaption><i>Figure 1: Visualisation of river rusting and iron mobilisation detected via satellite imagery (O'Donell et al., 2024).</i></figcaption>
</figure>

### Motivation
There is a need for a larger database of rusting rivers, to study this phenomenon and better monitor the associated environmental risks. Most existing detection approaches rely on local case studies, RGB visual interpretation and manually selected spectral bands. Improved methods would benefit from integrating high-dimensional contextual information to scale efficiently to continental monitoring. O'Donell et al. (2024) produced timelines of rusting change for three rivers which are used in this project to train an unsupervised learning model. Google's AlphaEarth satellite embeddings are used in this project. Embeddings encode spectral structure and contextual landscape information from lots of satellite products into multi-band latent representations that can be analysed statistically. This enables the design of a large scale detection framework.

This repository explores whether rusting rivers exhibit coherent trajectories in embedding space that can be learned and generalised, using
* pixel-level statistical analysis
* unsupervised learning using Principal Component Analysis (PCA) and Singular Value Decomposition (SVD)
* projection onto a learned universal rusting axis

The goal of this project is to build an Arctic-scale rusting river monitoring and detection framework capable of identifying anomalous river colour trajectories over time.


### Conceptual Framework

The framework treats river rusting as a directional trajectory in embedding space.

Instead of classifying rivers directly from imagery, the workflow:

1. extracts pixel-level embeddings from rusting rivers
2. compares embedding states between 2017 and 2020, the biggest change observed in timeline (Figure 1)
4. computes temporal displacement vectors
5. learns dominant rusting directions using PCA/SVD
6. projects unseen rivers into this learned change space

The central hypothesis is:

> Rusting rivers occupy a coherent directional manifold in embedding-change space that differs from normal temporal variability.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Workflows

### Data Extraction and Filtering in Google Earth Engine (GEE)

```mermaid
flowchart LR

B[Visual identification<br>of rusting reaches with Sentinel-2 imagery<br>2017 & 2020]
B --> D[Manual river polygon<br>delineation in GEE]


E[AlphaEarth annual embeddings] --> F[Apply river masks]
D --> F

F --> G[Stack yearly 64-band embeddings]
G --> H[Export GeoTIFFs to Google Drive]
```

### Analysis and Rusting Detection (Python / Colab)

```mermaid
flowchart LR

A[Load data and examine mean embedding timeline]
A --> B[Compute temporal embedding deltas] --> J[Identify layers with most change]
A --> D[PCA learning on balance pixel sampling for 2017 & 2020]

D --> E[Construct rusting axis<br>via SVD]
E --> F[Project unseen rivers<br>into PCA space]

F --> H[Scoring with cosine<br>alignment to rust axis]

```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### 1. Data Sources and Extraction Pipeline

#### Sentinel-2 Imagery for Rusting Delineation

River polygons were manually digitised in GEE. Sentinel-2 surface reflectance imagery (`COPERNICUS/S2_HARMONIZED`) was first used within GEE to visually identify and constrain river sections exhibiting rusting behaviour between 2017 and 2020.

For each study river:

- cloud-minimised summer composites were generated for both 2017 and 2020
- RGB visualisations (`B4`, `B3`, `B2`) were compared interactively
- river reaches exhibiting visible orange-brown discolouration were manually delineated using polygon masks

These masks were intentionally constrained to the visibly affected river pixels to  provide spatially targeted training data for embedding-space analysis. This minimises inclusion of unaffected reaches, reduces mixed land-water pixels and isolates regions undergoing observable optical change.


#### AlphaEarth Embeddings

Embeddings are obtained within GEE from:

`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`

Each pixel contains a 64-dimensional representation encoding spatial and spectral structure.
Embedding data was masked and clipped to the river polygons to isolate the water surface, before being exported as a individual GeoTiff files for further analysis (64 layers & 9 years for every pixel).



### Interpretation of the Approach

This workflow implements a **data-driven embedding analysis pipeline** where river rusting is treated as a directional shift in high-dimensional feature space.

Unlike classification-based approaches, this method:

- operates directly on temporal embedding differences
- does not require explicit labels for rusting intensity
- uses PCA as a shared representation space
- defines rusting as a dominant vector of change rather than a binary class

### Limitations

- Rust axis is derived from observed rivers only
- Potential sensitivity to sampling variability
- Controls are identified from satellite images only, ground truthing would be ideal.

### Planned Extensions

Future work should include:

* use of Arctic-wide river-network shapefiles
* automated river polygon generation, clipping shapefile to a 20km grid
* extraction of embedding for each clipped river segment
* projection of embeddings in the PCA space learned in this notebook.
* computation of probabilistic rusting likelihood scores
* spatial clustering of rusting hotspots
* further work on explaining embedding decomposition

By projecting embeddings of candidate rivers to the rusting axis learned in this project, rusting rivers can be identified at large scale. This framework would benefit from additional ground data to train the PCA model on additional rusting timelines.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

---
# Getting Started

This project was created using [Google Colab]([https://colab.research.google.com/]), a cloud-based platform for modifying and sharing code. To get a copy of this project up and running, follow these steps:
1. Download `Rusting_Rivers.ipynb` from this repository to get started.
2. Download datasets to use. See below for how to acquire datasets.
3. Modify file paths in the Colab notebook to match the location where the dataset was saved. 
4. Run the cells of the notebook sequentially. Further instructions can be found in the notebook.


### Downloading Datasets
The preliminary data extraction is documented in GEE-compatible .json files. Alpha Earth Embeddings from the Agashashuk, Kugururok and Anaktok rivers have been included in the repository for eaes of use, and can be directly used to run the colab notebook. Download these files in google drive, for which you can get a free account.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
# License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
# Contact

Benjamin Guillaud-Leblanc - [ben.guillaud-leblanc.22@ucl.ac.uk](mailto:ben.guillaud-leblanc@ucl.ac.uk) / [ben.guillaud@protonmail.com](mailto:ben.guillaud@protonmail.com) / [www.linkedin.com/in/benjamin-guillaud-leblanc](http://www.linkedin.com/in/benjamin-guillaud-leblanc)

Project Link: https://github.com/ben-g26/GEOL0069-Rusting-Rivers

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
# Acknowledgments
This project was created for GEOL0069 at University College London, taught by Dr. Michel Tsamados and Weibin Chen.
Thank you to Dr. Alexander Lipp for guidance during this project.


## References
O’Donnell, J. A., Carey, M. P., Koch, J. C., Baughman, C., Hill, K., Zimmerman, C. E., ... & Poulin, B. A. (2024). Metal mobilization from thawing permafrost to aquatic ecosystems is driving rusting of Arctic streams. *Communications Earth & Environment*, 5(1), 268.
https://doi.org/10.1038/s43247-024-01446-z
