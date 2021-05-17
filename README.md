# License-Recognition-SSD
Bài này sẽ giới thiệu phần License Recognition:

    Ở đây mình dùng MobileNet V2 SSD Lite model

Dependencies

    pytorch
    numpy
    opencv
    
Quick start

- Các bạn có thể clone repo này về.
- Tạo thư mục models và Download model theo file model.txt về thư mục vừa tạo
- Tạo thư mục images/images_long/test và copy hình ảnh muốn test vào thư mục đó
- Chạy câu lệnh sau: python run_ssd_example.py mb2-ssd-lite  models/license_model.pth models/labels.txt images


Train:

python train_ssd.py --dataset-type voc  --datasets data/voc_plate_ocr_dataset/  --net mb2-ssd-lite --base-net mb2-imagenet-71_8.pth  --scheduler cosine --lr 0.01 --t-max 100 --validation-epochs 5 --num-epochs 150 

Results:

Project nhận diện biển số xe này của mình có thể hoạt động trên cả biển một dòng hoặc hai dòng. Phần code run_ssd_example.py đã có phần sort để sắp xếp kết quả show ra terminal đúng với thứ tự của biển số.

[![Watch the video](https://img.youtube.com/vi/aZ0yFEe8_c8/0.jpg)](https://www.youtube.com/watch?v=aZ0yFEe8_c8)
