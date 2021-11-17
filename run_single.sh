NETWORKID=$1

python -m src.get_edgelist single $NETWORKID
python -m src.get_samples single $NETWORKID
python -m src.get_features single /data/s1620444/$NETWORKID
python -m src.stats single $NETWORKID
python -m src.get_performance single-all-features $NETWORKID

