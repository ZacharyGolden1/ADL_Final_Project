!#/bin/sh

cd data/Agriculture-Vision-2021

for set in train val test; do
    cd $set
    ls images/rgb | cut -d'.' -f1 > data.csv
    cd ..
done
