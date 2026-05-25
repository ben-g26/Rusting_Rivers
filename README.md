<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<img width="1546" height="697" alt="image" src="https://github.com/user-attachments/assets/87c62aff-8b45-4690-bcee-a788afd9f5f4" />

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
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#background">Background</a></li>
        <li><a href="#motivation">Motivation</a></li>
        <li><a href="#conceptual-framework">Conceptual Framework</a></li>
        <li><a href="#main-workflow">Main Workflow</a></li>
        <li>
          <a href="#data-sources-and-extraction-pipeline">Data Sources and Extraction Pipeline</a>
          <ul>
            <li><a href="#sentinel-2-imagery-for-rusting-delineation">Sentinel-2 Imagery for Rusting Delineation</a></li>
            <li><a href="#alphaearth-embeddings">AlphaEarth Embeddings</a></li>
          </ul>
        </li>
        <li>
          <a href="#principal-component-analysis-pca">Principal Component Analysis (PCA)</a>
          <ul>
            <li><a href="#limitations">Limitations</a></li>
            <li><a href="#planned-extensions">Planned Extensions</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#downloading-datasets">Downloading Datasets</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li>
      <a href="#acknowledgments">Acknowledgments</a>
      <ul>
        <li><a href="#references">References</a></li>
      </ul>
    </li>
  </ol>
</details>

# About the Project
This project is a final assignment for GEOL0069 at University College London, created to explore the usage of machine learning (ML) algorithms in Earth Sciences applications. It presents a scalable framework for detecting and characterising river rusting across Arctic environments using Google's Alpha Earth machine learning embeddings. The unsupervised learning method used to intepret the data is Principal Component Analysis (PCA).

## Background
The approach is inspired by the findings of O'Donell et al. (2024), [*"Metal mobilization from thawing permafrost to aquatic ecosystems is driving rusting of Arctic streams"*](https://www.nature.com/articles/s43247-024-01446-z#auth-Jonathan_A_-O_Donnell-Aff1), which identified widespread river colour changes linked to permafrost thaw in Alaska. Acid rock drainage conditions leading to iron and toxic heavy metal mobilisation are observed as previously frozen soil is exposed to an oxidizing environment. Iron-rich groundwater released from thawing permafrost alters river optical properties, producing strong orange-brown discolouration detectable from satellite imagery. Timeseries from 3 rusting rivers were compiled by the authors, and used in this project to train an unsupervised learning detection model.

<figure>
  <img width="1735" height="378" alt="Satellite imagery showing river rusting" src="https://github.com/user-attachments/assets/a124c0f9-1dbf-4ae2-8c6e-40712118ff16" />
  <figcaption><i>Figure 1: Visualisation of river rusting and iron mobilisation detected via satellite imagery (O'Donell et al., 2024).</i></figcaption>
</figure> <br><br/>

The mechanism behind the change in river colour is illustrated in Figure 2. As permafrost thaws by active layer thickening, sulfide minerals such as pyrite are exposed to chemical weathering, releasing sulfate, acid, and trace metals into groundwater and streams.
<figure>
 <img width="1566" height="497" alt="image" src="https://github.com/user-attachments/assets/94caaf10-67d7-41f6-a7b9-4715afad3fd2" />
  <figcaption><i>Figure 2: Mechanism of river rusting from thawing permafrost (O'Donell et al., 2024).</i></figcaption>
</figure> <br><br/>

River rusting has direct impacts on riverine foodchains and human water use. The main environmental effects are detailed in Figure 3.
<figure>
 <img width="1693" height="472" alt="image" src="https://github.com/user-attachments/assets/7e1a28d4-8b4c-42b8-891e-ae0e69a158f4" />
  <figcaption><i>Figure 3: Impacts of river rusting on humans and foodchains (O'Donell et al., 2024).</i></figcaption>
</figure>



This chemical change is altering river ecosystems in many ways
### Motivation
There is a need for a larger database of rusting rivers, to study this phenomenon and better monitor the associated environmental risks. Most existing detection approaches rely on local case studies, RGB visual interpretation and manually selected spectral bands. Improved methods would benefit from integrating high-dimensional contextual information to scale efficiently to continental monitoring. O'Donell et al. (2024) produced timelines of rusting change for three rivers which are used in this project to train an unsupervised learning model. Google's AlphaEarth satellite embeddings provide ready-made high-dimensional data at 10m resolution to train such a model. Embeddings encode spectral structure and contextual landscape information from a large range of satellite products into multi-band latent representations that can be analysed statistically. This enables the design of a large scale detection framework.

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

## Main Workflow

<figure>
<img width="1536" height="920" alt="image" src="https://github.com/user-attachments/assets/4b2f7c38-892b-495c-912e-b96406eaf9dd" />
  <figcaption><i>Figure 4: Data processing workflow.</i></figcaption>
</figure> <br><br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Data Sources and Extraction Pipeline

### Sentinel-2 Imagery for Rusting Delineation

River polygons were manually digitised in Google Earth Engine (GEE). Sentinel-2 surface reflectance imagery (`COPERNICUS/S2_HARMONIZED`) was first used within GEE to visually identify and constrain river sections exhibiting rusting behaviour between 2017 and 2020.

For each study river:

- cloud-minimised summer composites were generated for both 2017 and 2020
- RGB visualisations (`B4`, `B3`, `B2`) were compared interactively
- river reaches exhibiting visible orange-brown discolouration were manually delineated using polygon masks

These masks were intentionally constrained to the visibly affected river pixels to  provide spatially targeted training data for embedding-space analysis. This minimises inclusion of unaffected reaches, reduces mixed land-water pixels and isolates regions undergoing observable optical change.


### AlphaEarth Embeddings

Embeddings are obtained within GEE from:

`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`

Embeddings are extracted for rusting rivers and their non-rusting control. Controls are nearby river segments not affected by rusting:
<figure>
<img width="521" height="339" alt="image" src="https://github.com/user-attachments/assets/cef54059-dc92-46df-8461-1980cf8dc3c5" />
    <br>
  <figcaption><i>Figure 5: Agashashok river shapefiles for rusting river section (blue) and the upstream non-affected control (red).</i></figcaption>
</figure> <br><br/>

Each pixel contains a 64-dimensional representation encoding spatial and spectral structure.
Embedding data was masked and clipped to the river polygons to isolate the water surface, before being exported as a individual GeoTiff files for further analysis (64 layers & 9 years for every pixel).



## Principal Component Analysis (PCA)

This workflow implements a **data-driven embedding analysis pipeline** where river rusting is treated as a directional shift in high-dimensional feature space. <br>

Principal Component Analysis (PCA) is used to identify the dominant modes of variability within the AlphaEarth embeddings, which encode each pixel as a 64-dimensional vector representing spectral, spatial and contextual information. Rather than performing PCA on temporal differences directly, the method first combines balanced samples of 2017 and 2020 embeddings from the training rusting rivers into a shared embedding dataset. PCA then learns an orthogonal coordinate system that captures the major axes of variation across these embedding states. Each pixel embedding from both years is projected into this reduced PCA space, allowing temporal change vectors to be computed after projection. In this representation, river rusting can be interpreted as a coherent directional trajectory through embedding space rather than a simple spectral anomaly. The leading components therefore capture structured environmental variation, while the resulting change vectors reveal whether different rivers evolve along a similar “rusting direction”. This enables comparison of temporal behaviour across rivers with different sizes, locations and baseline spectral characteristics, while reducing noise and redundancy present in the original 64-dimensional embeddings.

### Limitations

- Rust axis is derived from a limited number of rivers
- There is potential sensitivity to sampling variability
- Controls are identified from satellite images only, ground truthing of non-rusting rivers would be ideal.

## Planned Extensions
By projecting embeddings of candidate rivers to the rusting axis learned in this project, rusting rivers can be identified at large scale. Future work should first refine a universal rusting axis in 64-D space using additional rusting river timelines to train the model. Upscaling this framework to the entire arctic can then be done with:
* use of Arctic-wide river-network shapefiles
* automated river polygon generation, clipping shapefiles to a 20km grid
* extraction of embedding for each clipped river segment
* projection of embeddings in the PCA space learned in this notebook.
* computation of probabilistic rusting likelihood scores
Such a dataset could be used to measure direct impacts of climate change on the Arctic and its ecosystems, as well as constrain factors that contribute to river rusting (geology, landscape, hydrogeology, etc.) 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---
# Getting Started

This project was created using [Google Colab]([https://colab.research.google.com/]), a cloud-based platform for modifying and sharing code. To get a copy of this project up and running, follow these steps:
1. Download `Rusting_Rivers.ipynb` from this repository to get started.
2. Download datasets to use. See below for how to acquire datasets.
3. Modify file paths in the Colab notebook to match the location where the dataset was saved. 
4. Run the cells of the notebook sequentially. Further instructions can be found in the notebook.


### Downloading Datasets
The preliminary data extraction is documented in GEE_Extraction_Pipeline.json. Alpha Earth Embeddings from the Agashashuk, Kugururok and Anaktok rivers as well as nearby control rivers have been included in the repository for eaes of use, and can be directly used to run the colab notebook. Download these files in google drive, for which you can get a free account.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
# License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
# Contact

Benjamin Guillaud-Leblanc - [ben.guillaud-leblanc.22@ucl.ac.uk](mailto:ben.guillaud-leblanc@ucl.ac.uk) / [ben.guillaud@protonmail.com](mailto:ben.guillaud@protonmail.com) / [www.linkedin.com/in/benjamin-guillaud-leblanc](https://linkedin.com/in/benjamin-guillaud-leblanc-b168b723a)

Project Link: https://github.com/ben-g26/GEOL0069-Rusting-Rivers

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
# Acknowledgments
This project was created for GEOL0069 at University College London, taught by Dr. Michel Tsamados and Weibin Chen.
Thank you to Dr. Alexander Lipp for guidance during this project.


## References
O’Donnell, J. A., Carey, M. P., Koch, J. C., Baughman, C., Hill, K., Zimmerman, C. E., ... & Poulin, B. A. (2024). Metal mobilization from thawing permafrost to aquatic ecosystems is driving rusting of Arctic streams. *Communications Earth & Environment*, 5(1), 268.
https://doi.org/10.1038/s43247-024-01446-z
