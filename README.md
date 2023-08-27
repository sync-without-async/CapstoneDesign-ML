# ReHab Machine Learning - Pose Estimation

본 저장소에서는 ReHab 프로젝트의 인공지능 저장소 입니다. 프로젝트에서 인공지능은 핵심적인 요소 입니다. 사용자에게 운동을 어떻게 할 수 있도록 제공해주는 핵심적인 서비스이자 방법을 인공지능이 제공하기 때문입니다. 우리는 이 인공지능을 통해서 가이드 영상과 사용자에게 입력 받은 영상을 Feature Extraction 후 Cosine Similarty를 통해서 유사도 측정을 통해서 얼마나 잘 되고 있는지를 확인합니다.

모델은 Pre-trained된 모델을 사용하였습니다. Baseline Model은 Posenet을 활용하고 있으며 Communication Overhead와 Computation Overhead의 Trade-off 등을 통해서 이는 변경될 수 있습니다.

## Our Model

Pytorch의 `torchvision.models` 모듈에서는 여러 Pretrained 모델과 SOTA 모델 구조들을 제공합니다. 우리는 선행학습된 사람의 모습을 추출하는 것이 중요하므로 Keypoint 추출이 가능한 Keypoint RCNN ResNet50 FPN 기반 모델을 사용하였습니다. Keypoint R-CNN + ResNet50 FPN 의 구조라고 이해하면 쉬울 거 같네요.

[공식 문서](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn)에 따르면 default weight는 COCO Dataset v1로 학습된 모델로 기존 Legacy 모델보다 더 많은 Parameter를 지녔지만 GFLOPs는 떨어졌으며 성능도 개선된 모델입니다.
