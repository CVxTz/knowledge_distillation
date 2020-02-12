for i in {0..5}
do
python baseline_har.py > baseline_har.txt
python ensemble_har.py > ensemble_har.txt
#python permado_har.py > permado_har.txt
done