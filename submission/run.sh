WBCtrain="./test/WBC_data/wdbc-train-csv.data"
WBCtest="./test/WBC_data/wdbc-test-csv.data"
MPGtrain="./test/cars/train.csv"
MPGtest="./test/cars/test.csv"

Driver="./src/driver.py"

# mpg predict on training static
python $Driver --train $MPGtrain --test $MPGtrain -r -c static -b 3 -l 3 -d 6 -p Good
# mpg predict on training dynamic
python $Driver --train $MPGtrain --test $MPGtrain -r -c dynamic -b 3 -l 3 -d 6 -p Good
# mpg predict on testing static
python $Driver --train $MPGtrain --test $MPGtest -r -c static -b 3 -l 3 -d 6 -p Good --print
# mpg predict on testing dynamic
python $Driver --train $MPGtrain --test $MPGtest -r -c dynamic -b 3 -l 3 -d 6 -p Good

# wbc predict on training static
python $Driver --train $WBCtrain --test $WBCtrain -r -c static -b 3 -l 3 -d 6 -p 4
# wbc predict on training dynamic
python $Driver --train $WBCtrain --test $WBCtrain -r -c dynamic -b 3 -l 3 -d 6 -p 4
# wbc predict on testing static
python $Driver --train $WBCtrain --test $WBCtest -r -c static -b 3 -l 3 -d 6 -p 4
# wbc predict on testing dynamic
python $Driver --train $WBCtrain --test $WBCtest -r -c dynamic -b 3 -l 3 -d 6 -p 4

# kfold
python $Driver --train $WBCtrain --test $WBCtrain -r -c dynamic -b 3 -l 3 -d 6 -p 4 -k

# sklearn
python ./src/sk-test.py $WBCtrain
