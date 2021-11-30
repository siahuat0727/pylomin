for batch_size in 1 2 4 8 16
do
	for method in lazy-load+group-embedding lazy-load  naive lazy-load+keep-embedding
	do
		python3 main.py --batch_size $batch_size --method $method --use_gpu
		python3 main.py --batch_size $batch_size --method $method
	done
done
