for i in 40 41 42 43 44; do
    echo "Running seed $i"
    python main.py --dataset sub \
    --model kldistill \
    --n_epochs 500 --lr 0.005 --batch_size 512 --exp_decay 0.995 \
    --seed $i \
    --teacher_backbone shallow \
    --teacher_ckpt data/runs/sub-standard-cbm-epoch499-seed400.pt \
    --teacher cbm \
    --all_metrics

    python main.py --dataset sub \
    --model logitdistill \
    --loss_type l1 \
    --n_epochs 500 --lr 0.005 --batch_size 512 --exp_decay 0.995 \
    --seed $i \
    --teacher_backbone shallow \
    --teacher_ckpt data/runs/sub-standard-cbm-epoch499-seed400.pt \
    --teacher cbm \
    --all_metrics

    python main.py --dataset sub \
    --model logitdistill \
    --loss_type l2 \
    --n_epochs 500 --lr 0.005 --batch_size 512 --exp_decay 0.995 \
    --seed $i \
    --teacher_backbone shallow \
    --teacher_ckpt data/runs/sub-standard-cbm-epoch499-seed400.pt \
    --teacher cbm \
    --all_metrics
done
