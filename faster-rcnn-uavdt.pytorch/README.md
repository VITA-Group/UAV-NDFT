# UAV-AdversarialLearning
## Multiple GPU Training
```{r, engine='bash', count_lines}
#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python trainval_net_monitor.py --cuda --mGPUs --gamma 0.01 --monitor_discriminator --use_adversarial_loss --use_restarting --use_tfb --bs 4
```
## Single GPU Training
```{r, engine='bash', count_lines}
#!/bin/bash
for ((i=0; i<=10; i++))
do
        epoch=$(($i*1000/11914+1))
        ckpt=$((i*1000%11914))
        echo "$epoch"
        echo "$ckpt"
        CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda --checkepoch "$epoch" --checkpoint "$ckpt" --gamma 0.5
done

```
## UAVDT Data (Training+Testing) in Pascal Voc Format
Google Drive: https://drive.google.com/file/d/13xdLBfIWGYrjpT0Z3miAPKnKNDjNqLS9/view?usp=sharing

## UAVDT Trained Model (w/o Adversarial Loss and w/ Adversarial Loss)
Google Drive: https://drive.google.com/file/d/1rxqr0Cq0y9cXhdWyNd_R_8cd68exD1wn/view?usp=sharing


## Project Directory Layout
```
.
├── cfgs
├── data              # UAVDT dataset with annotation
├── images
├── lib
├── logs              # TensorBoard event files
├── models            # Trained model (w/ adversarial loss and w/o adversarial loss)
├── output
├── summaries         # Summary files recording the training and validation performance
├── README.md
├── _init_paths.py
├── bash_run.sh       # Run the testing in batch
├── demo.py
├── requirements.txt
├── test_net.py
├── trainval_net.py
└── trainval_net_monitor.py
```
