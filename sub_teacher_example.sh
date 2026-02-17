for i in 400 401 402 403 404; do
    echo "Running seed $i"
    python main.py --dataset sub \
    --model cbm \
    --n_epochs 500 --lr 0.005 --batch_size 512 --exp_decay 0.995 \
    --seed $i 
done
