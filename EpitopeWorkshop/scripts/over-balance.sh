set -x

source ~/miniconda3/etc/profile.d/conda.sh
conda activate epitope-workshop37
totalWorkers=$1
for ((i = 0; i < totalWorkers; i++))
do
python /Users/sfeiner/Documents/studies/biology/EpitopeWorkshop/EpitopeWorkshop/main.py \
over-balance-dir \
--features-df-pickle-path-dir /Users/sfeiner/Documents/studies/biology/EpitopeWorkshop/data/iedb-linear-epitopes-parts-2/features/ \
--total-workers $totalWorkers --worker-id $i &
done