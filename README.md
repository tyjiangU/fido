# FIDO
This is the code for our paper [Exploiting Definitions for Frame Identification](https://aclanthology.org/2021.eacl-main.206.pdf) (EACL 2021).

## Testing Environment
This project is built on python==3.7.6, torch=1.4.0, transformers==2.9.0.

## Use Our Pre-trained Models - FIDO
Download the pre-trained models at: [trained on FrameNet 1.5](https://drive.google.com/file/d/1NGDU_bysROH22vh_sa6XHSb10e_edGyE/view?usp=sharing), [trained on FrameNet 1.7](https://drive.google.com/file/d/149LgbwLfTq_uyKhaWRg7ZNme2H25Yihg/view?usp=sharing).

The accuracy of the pre-trained models are:
|      | dev | test |
| ---- | --- | ---- |
|FN 1.5| 92.4| 91.5 |
|FN 1.7| 92.4| 92.3 |

The file you would like to predict should be "data/fn1.5/test.csv" or "data/fn1.7/test.csv".

Put the extracted "model_fn1.5" or "model_fn1.7" folder under the "pretrained_models/" directory, and run predict.sh.

It will generate two files: "test_prediction_labels.txt" and "test_prediction_probs.txt" under the model directory.

## Train from Scratch
For FrameNet 1.5, data files (train.csv, dev.csv and test.csv) should be put under "data/fn1.5/".

For FrameNet 1.7, similarly, data files should be put under "data/fn1.7/".

Run train.sh. You should get similar results compared to the table above.

## Data Format
id, sentence, lu_name, lu_head_position, lu_defs, frame_names, frame_defs, label <br>

lu_name: the target word or phrase <br>
lu_head_position: the position index of the target in the sentence <br>
lu_defs: all the target word definitions associated with the candidate frames (each LU will have different definitions for different associated frames), separated by "\~$\~" <br>
frame_names: candidate frames, separated by "\~$\~" <br>
frame_defs: candidate frame definitions, separated by "\~$\~" <br>
label: an integer indicating the correct frame from the frame_names (lu_defs, frame_names and frame_defs should have the same corresponding order)

An example of the data format can be found under "data/fn1.5/".

The "data/fn1.5/" folder ***\*only contains a small sample\**** of the data. To replicate the results in the paper, you will need full text of the FrameNet data, as well as the same data split.

Get full access to the FrameNet data at [here](https://framenet.icsi.berkeley.edu/fndrupal/framenet_data).

We followed the same train/dev/test split as in [Das et al. (2014)](https://direct.mit.edu/coli/article/40/1/9/1461/Frame-Semantic-Parsing) and [Swayamdipta et al. (2017)](https://arxiv.org/abs/1706.09528). Details of the data processing can be found at [Open-SESAME](https://github.com/swabhs/open-sesame).

## Contact and Reference
For questions and issues, please contact `tianyu@cs.utah.edu`. Our paper can be cited as:
```
@inproceedings{jiang-riloff-2021-exploiting,
title="{Exploiting Definitions for Frame Identification}",
author={Jiang, Tianyu and Riloff, Ellen},
booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)},
year={2021}
}
```
