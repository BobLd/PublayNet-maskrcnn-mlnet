# PublayNet-maskrcnn-mlnet
Using a MaskRCNN model trained on the PublayNet dataset with [ML.Net](https://github.com/dotnet/machinelearning) in C# / .Net for Document layout analysis and page segmmentation task.

Recognised regions/categories are:
- Text (i.e. paragraph)
- Title
- List
- Table
- Figure

## About
- About the PublayNet dataset: [ibm-aur-nlp/PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)
- Original repos for training the model: [phamquiluan/PubLayNet](https://github.com/phamquiluan/PubLayNet)
- Original MaskRCNN-Resnet50-FPN checkpoint [here](https://drive.google.com/file/d/1Jx2m_2I1d9PYzFRQ4gl82xQa-G7Vsnsl/view?usp=sharing)
- Jupyter notebook to convert for ONNX model are [here](https://github.com/BobLd/PublayNet-maskrcnn-mlnet/tree/master/PublayNetModelTEst/notebooks)
- Compressed ONNX model in [PublayNetModelTEst/Assets/Model](https://github.com/BobLd/PublayNet-maskrcnn-mlnet/tree/master/PublayNetModelTEst/Assets/Model)

## Results
See [here](https://github.com/BobLd/PublayNet-maskrcnn-mlnet/tree/master/PublayNetModelTEst/Assets/Output)

![result_1](https://github.com/BobLd/PublayNet-maskrcnn-mlnet/blob/master/PublayNetModelTEst/Assets/Output/PMC5055614_00000.jpg)
![result_2](https://github.com/BobLd/PublayNet-maskrcnn-mlnet/blob/master/PublayNetModelTEst/Assets/Output/PMC5055614_00002.jpg)
![result_3](https://github.com/BobLd/PublayNet-maskrcnn-mlnet/blob/master/PublayNetModelTEst/Assets/Output/foo.0_raw.png)
