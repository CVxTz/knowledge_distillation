for i in {0..5}
do
python small_ecg.py > small_ecg.txt
python baseline_ecg.py > baseline_ecg.txt
python distillation_ecg.py > distillation_ecg.txt
done