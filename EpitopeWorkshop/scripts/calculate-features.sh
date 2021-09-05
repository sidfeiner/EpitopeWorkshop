source ~/miniconda3/etc/profile.d/conda.sh
conda activate epitope-workshop37
totalWorkers=$1
for ((i = 0; i < totalWorkers; i++))
do
python /Users/sfeiner/Documents/studies/biology/EpitopeWorkshop/EpitopeWorkshop/main.py \
calculate-features-dir --total-workers $totalWorkers --worker-id $i \
--sequences_files_dir /Users/sfeiner/Documents/studies/biology/EpitopeWorkshop/data/iedb-linear-epitopes-parts-2 &
done