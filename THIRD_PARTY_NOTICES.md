# Third-party notices and attribution

Jormungandr uses and builds on the following datasets, software, and research.
The project code and shipped ONNX artifacts are distributed under GPL-3.0-only;
third-party materials remain subject to their own terms.

## STEAD

The student model was trained on the **STanford EArthquake Dataset (STEAD)**,
and the streaming demonstration plot contains transformed excerpts from STEAD.
STEAD is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The waveforms used here were demeaned, bandpass-filtered, normalized, and used to
train or demonstrate this model.

> Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019). STanford
> EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI.
> *IEEE Access*. https://doi.org/10.1109/ACCESS.2019.2947848

Source: https://github.com/smousavi05/STEAD

## EQTransformer

Knowledge distillation used outputs from the pretrained EQTransformer `stead`
teacher distributed through SeisBench. The original EQTransformer project is
MIT-licensed.

> Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C.
> (2020). Earthquake transformer—an attentive deep-learning model for
> simultaneous earthquake detection and phase picking. *Nature Communications,
> 11*, 3952. https://doi.org/10.1038/s41467-020-17591-w

Source: https://github.com/smousavi05/EQTransformer

## SeisBench

STEAD access and the pretrained EQTransformer implementation/weights were
provided through SeisBench 0.11.7, licensed under GPLv3. The cached pretrained
model metadata requests citation of the SeisBench publications.

> Woollam, J., Münchmeyer, J., Tilmann, F., et al. (2022). SeisBench—A Toolbox
> for Machine Learning in Seismology. *Seismological Research Letters*.
> https://doi.org/10.1785/0220210324

Source: https://github.com/seisbench/seisbench

## Other dependencies

PyTorch, NumPy, SciPy, Matplotlib, PyYAML, ONNX, ONNX Runtime, and pytest are
used under their respective upstream licenses. See `requirements.txt` for the
exact versions used to verify the published results.
