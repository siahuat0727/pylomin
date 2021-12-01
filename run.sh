for batch_size in 1 2 4 8 16
do
	for method in lazy-loading+chunked-embedding lazy-loading  naive lazy-loading+keep-embedding
	do
		# Run on GPU to trace the peak memory needed for the inference
		python3 demo.py --batch_size $batch_size --method $method --use_gpu
		python3 demo.py --batch_size $batch_size --method $method
	done
done
