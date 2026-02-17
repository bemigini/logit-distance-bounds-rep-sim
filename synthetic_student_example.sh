for i in 40 41 42 43 44; do
    echo "Running seed $i"
    python main.py --dataset synthetic \
    --model logitdistill \
    --loss_type l1 \
    --n_epochs 250 \
    --lr 0.001 \
    --batch_size 512 --exp_decay 0.995 \
    --seed $i \
    --teacher_backbone mlp \
    --teacher_ckpt  data/runs/synthetic-standard-classifier-epoch499-seed400.pt \
    --teacher classifier \
    --modality standard \
    --all_metrics
done