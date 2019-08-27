echo "downloading the ransomware classification dataset..."

MODEL=./data.zip
URL_MODEL=https://data.ncl.ac.uk/ndownloader/files/17434379

echo "downloading the dataset..."

wget --quiet --no-check-certificate --show-progress $URL_MODEL -O $MODEL

echo "checking the MD5 checksum for downloaded file..."

CHECK_SUM_CHECKPOINTS='e955db2cf2bc11a3e79e2d2153299fce  pre_trained_weights.zip'

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the zip file..."

unzip -q pre_trained_weights.zip && rm pre_trained_weights.zip 

cd pre_trained_weights

rm README.txt

echo "All Done!!"
