if [ ! -d ./data/ ]; then
  mkdir -p ./data;
fi

wget http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

mv HIGGS.csv.gz ./data 