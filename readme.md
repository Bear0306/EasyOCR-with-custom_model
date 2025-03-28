# Instruction(Ubuntu)

### 1. Shell cmd(environment construct)

root@ip-172-31-19-60:/home/ubuntu# sudo apt update && sudo apt upgrade -y

root@ip-172-31-19-60:/home/ubuntu# sudo apt install python3-pip -y

root@ip-172-31-19-60:/home/ubuntu# apt install python3-fire -y

root@ip-172-31-19-60:/home/ubuntu# apt install python3-lmdb -y

root@ip-172-31-19-60:/home/ubuntu# apt install python3-opencv -y

root@ip-172-31-19-60:/home/ubuntu# apt install python3-natsort -y

root@ip-172-31-19-60:/home/ubuntu# apt install python3-nltk -y

root@ip-172-31-19-60:/home/ubuntu# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages

root@ip-172-31-19-60:/home/ubuntu# git clone https://github.com/clovaai/deep-text-recognition-benchmark

root@ip-172-31-19-60:/home/ubuntu# apt install mc -y

root@ip-172-31-19-60:/home/ubuntu# cd deep-text-recognition-benchmark

### 2. Create lmdb dataset

"Download images and label.txt into /deep-text-recognition-benchmark/output/"

root@ip-172-31-19-60:/home/ubuntu/deep-text-recognition-benchmark# python3 ./create_lmdb_dataset.py ./output ./output/labels.txt ./lmdb_output

### 3. Train

"Download and add TPS-ResNet-BiLSTM-CTC.pth into /deep-text-recognition-benchmark/"

"Replace /deep-text-recognition-benchmark/dataset.py"

"Add /deep-text-recognition-benchmark/train_cpu.py"

root@ip-172-31-19-60:/home/ubuntu/deep-text-recognition-benchmark# python3 train_cpu.py --train_data lmdb_output --valid_data lmdb_output --select_data "/" --batch_ratio 1.0 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --batch_size 2 --data_filtering_off --workers 0 --batch_max_length 80 --num_iter 10 --valInterval 5 --saved_model TPS-ResNet-BiLSTM-CTC.pth

### 4. EasyOcr

"Copy best_accuracy.pth from /deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/ to /root/.EasyOCR/model/"

"Copy modules, best_accuracy.py, bestaccuracy.yaml to /root/.EasyOCR/user_network/"

"Copy images to /deep-text-recognition-benchmark/easyocr_examples/"

root@ip-172-31-19-60:/home/ubuntu/deep-text-recognition-benchmark# pip install easyocr --break-system-packages

root@ip-172-31-19-60:/home/ubuntu/deep-text-recognition-benchmark# cd easyocr_example
root@ip-172-31-19-60:/home/ubuntu/deep-text-recognition-benchmark/easyocr_examples# python3
>>>import easyocr
>>>reader = easyocr.Reader(['en'], recog_network='best_accuracy')
>>>result = reader.readtext('chinese.jpg')
>>>reader.readtext('chinese.jpg', detail = 0)
