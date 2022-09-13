mkdir -p ./data/ISRUC_S3/ExtractedChannels
mkdir -p ./data/ISRUC_S3/RawData
echo 'Make data dir: ./data/ISRUC_S3'

cd ./data/ISRUC_S3/RawData

for i in `seq 1 10` #seq是一个命令，顺序生成一串数字或者字符
do
#    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$i.rar
    unrar x $i.rar
done

echo 'Download Data to "./data/ISRUC_S3/RawData" complete.'

cd ./data/ISRUC_S3/ExtractedChannels
for i in `seq 1 10`
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$i.mat
done
echo 'Download ExtractedChannels to "./data/ISRUC_S3/ExtractedChannels" complete.'
