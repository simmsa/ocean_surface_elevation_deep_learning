# DTSA 5511 Introduction to Machine Learning: Deep Learning
Andrew Simms
2024-12-09

# Overview

Ocean wave prediction is used for maritime safety, wave energy
conversion, and coastal engineering applications. This research explores
deep learning approaches for predicting ocean surface elevation
time-series using historical buoy measurements. While traditional wave
analysis relies on [statistical
parameters](https://www.coastalwiki.org/wiki/Statistical_description_of_wave_parameters)
including significant wave height ($H_{m_0}$) and energy period ($T_e$),
many applications could benefit from more accurate wave-by-wave
predictions of surface elevation.

The need for accurate wave prediction is particularly evident in [wave
energy](https://tethys.pnnl.gov/technology/wave) applications, where
Ringwood (2020) highlights challenges in control system optimization
that depend on reliable wave forecasts. Abdelkhalik et al. (2016)
demonstrates how wave predictions enable real-time optimization of
energy extraction, showing that accurate forecasting directly impacts
system performance and efficiency.

This project addresses the fundamental need for accurate near real-time
wave prediction by developing deep learning models to forecast
three-dimensional surface elevation time-series, focusing on maintaining
both prediction accuracy and computational efficiency through models
trained on previously collected measurements.

The full report can be viewed
[here](http://www.andrewdsimms.com/ocean_surface_elevation_deep_learning/).

# Model Code

Models and training code are in `./train_window_from_spec.py`

Report source code is in `./quarto/index.qmd`. Rendered documents
`./final_report.ipynb` (image rendering is not working),
`./final_roport.pdf` (image rendering works) and the final output can be
viewed
[here](http://www.andrewdsimms.com/ocean_surface_elevation_deep_learning/).

# Running Models

    ./batch_train.sh

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-abd_2016" class="csl-entry">

Abdelkhalik, Ossama, Rush Robinett, Shangyan Zou, Giorgio Bacelli, Ryan
Coe, Diana Bull, David Wilson, and Umesh Korde. 2016. “On the Control
Design of Wave Energy Converters with Wave Prediction.” *Journal of
Ocean Engineering and Marine Energy* 2 (4): 473–83.
<https://doi.org/10.1007/s40722-016-0048-4>.

</div>

<div id="ref-ringwood_2020" class="csl-entry">

Ringwood, John V. 2020. “Wave Energy Control: Status and Perspectives
2020 ⁎⁎This Paper Is Based Upon Work Supported by Science Foundation
Ireland Under Grant No. 13/IA/1886 and Grant No. 12/RC/2302 for the
Marine Renewable Ireland (MaREI) Centre.” *IFAC-PapersOnLine* 53 (2):
12271–82. <https://doi.org/10.1016/j.ifacol.2020.12.1162>.

</div>

</div>
