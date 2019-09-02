echo "downloading the ransomware classification dataset..."

MODEL=./dataset.zip
URL_MODEL=https://data.ncl.ac.uk/ndownloader/files/17434379

echo "downloading the dataset..."

wget --quiet --no-check-certificate --show-progress $URL_MODEL -O $MODEL

echo "checking the MD5 checksum for the downloaded file..."

CHECK_SUM_CHECKPOINTS='fd2cdfc012c78ab4b93e3fdbdc47df5c  dataset.zip'

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the zip file..."

unzip -q dataset.zip && rm dataset.zip && mv ransom_ware/* dataset && rm -r ransom_ware

echo "All Done!!"
