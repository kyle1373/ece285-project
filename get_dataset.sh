wget -c --show-progress https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0TrainPart1.zip -O Gnt1.0TrainPart1.zip
wget -c --show-progress https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0Test.zip -O Gnt1.0Test.zip

mkdir dataset_gnt

unzip Gnt1.0TrainPart1.zip -d dataset/
unzip Gnt1.0Test.zip -d dataset/

rm Gnt1.0TrainPart1.zip
rm Gnt1.0Test.zip