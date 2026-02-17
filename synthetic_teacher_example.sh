for i in 400 401 402 403 404; do
    echo "Running seed $i"
    python main.py --dataset synthetic \
    --model classifier \
    --n_epochs 500 \
    --lr 0.001 \
    --batch_size 512 --exp_decay 0.995 \
    --seed $i \
    --modality standard
done