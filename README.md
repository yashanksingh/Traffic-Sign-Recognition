<h1>Traffic Sign Recognition using YOLOv8 Algorithm
extended with CNN</h1>

<h3>This project uses a two-stage implementation for traffic sign recognition. On the first
stage, real-time video stream from the cameras is processed by the trained YOLO
model. Results are processed and bounding boxes are drawn around detections with
confidence over a pre-defined threshold. On the second stage, these detections are
cropped and are further processed by the trained CNN model which classifies the
traffic signs into 43 categories.</h3>


<p>
Datasets Used:<br>
&ensp; GTSRB - German Traffic Sign Recognition Benchmark<br>
&ensp; GTSDB - German Traffic Sign Detection Benchmark<br>
</p>

<p>
For training using GPU (tested with RTX3050 Mobile):<br>
&ensp; CUDA Toolkit v11.2.0<br>
&ensp; cuDNN v8.1.0<br>
</p>


[Link to Research Paper](assets/Traffic%20Sign%20Recognition%20Research%20Paper%20PBL%20v3.pdf)
