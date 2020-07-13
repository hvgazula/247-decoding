import os


def create_script(job_name_str, s_list):
    file_name = os.path.join(os.getcwd(), 'slurm_scripts',
                             f'{job_name_str}.sh')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w+') as fh:
        fh.write("#!/bin/bash\n")
        fh.write(f"#SBATCH --job-name={job_name_str}\n")
        fh.write("#SBATCH --output=./slurm_logs/%j-%x.out\n")
        fh.write("#SBATCH --error=./slurm_logs/%j-%x.err\n")
        fh.write("#SBATCH --nodes=1 #nodes\n")
        fh.write("#SBATCH --ntasks-per-node=1\n")
        fh.write("#SBATCH --cpus-per-task=4\n")
        fh.write("#SBATCH --mem=16G\n")
        fh.write("#SBATCH --time=0-02:00:00\n")
        fh.write("#SBATCH --gres=gpu:1\n")
        fh.write("#SBATCH --mail-type=begin\n")
        fh.write("#SBATCH --mail-type=fail\n")
        fh.write("#SBATCH --mail-type=end\n")
        fh.write("#SBATCH --mail-user=hvgazula@umich.edu\n")
        fh.write("\n")
        fh.write("module purge\n")
        fh.write("module load anaconda3\n")
        fh.write("conda activate torch-env\n")
        fh.write("\n")
        fh.write("if [[ -v $SLURM_TASK_ARRAY_ID ]]\n")
        fh.write("then\n")
        fh.write("\tSEED=$SLURM_TASK_ARRAY_ID\n")
        fh.write("else\n")
        fh.write("\tSEED=1234\n")
        fh.write("fi\n")
        fh.write(f"python {os.path.join(os.getcwd(), 'main.py')} \\\n")
        for item in s_list:
            fh.write(f'\t{item} \\\n')
        fh.write("\t--seed $SEED \\\n")
        fh.write(f"\t--output-folder {job_name_str}\n")

    return file_name


# model = ["MeNTAL"]
# subjects = [625, 676]
# max_electrodes = [55, 64]
# shift = [25, 50, 100, 150, 250, 500]
# lr = 1e-4
# gpus = 1
# epochs = 100
# batch_size = 120
# bin_size = [25, 50, 75, 100, 125, 150]
# tf_weight_decay = [0.01, 0.025, 0.05, 0.075]
# tf_dropout = [0.1]
# tf_nlayers = [3, 6, 9, 12]
# tf_nheads = [4, 8, 16]
# tf_dmodel = [128, 256, 512, 1024]
# tf_dff = [128, 256, 512, 1024]
# temp = [0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]

model = ["MeNTAL"]
subjects = [625]
max_electrodes = [55]
shift = [0]
lr = 1e-4
gpus = 2
epochs = 100
batch_size = 120
bin_size = [50]
tf_weight_decay = [0.01]
tf_dropout = tf_weight_decay
tf_nlayer = [3]
tf_nhead = [4]
tf_dmodel = [128]
tf_dff = [128]
temp = [0.9]

str_lr = '--lr ' + str(lr)
str_gpu = '--gpus ' + str(gpus)
str_epochs = '--epochs ' + str(epochs)
str_batch_size = '--batch-size ' + str(batch_size)

# TODO: Poorly written for loops, needs refactoring
# TODO: Code to run for combination of subjects
count = 0
s = [str_lr, str_gpu, str_epochs, str_batch_size]
for a in model:
    s1 = '--model ' + str(a)
    for b in subjects:
        s2 = '--subjects ' + str(b)
        for c in max_electrodes:
            s3 = '--max-electrodes ' + str(c)
            for d in shift:
                s4 = '--shift ' + str(d)
                for e in bin_size:
                    s5 = '--bin-size ' + str(e)
                    for f in tf_weight_decay:
                        s6 = '--weight-decay ' + str(f)
                        for g in tf_dropout:
                            s7 = '--tf-dropout ' + str(g)
                            for h in tf_nlayer:
                                s8 = '--tf-nlayer ' + str(h)
                                for i in tf_nhead:
                                    s9 = '--tf-nhead ' + str(i)
                                    for j in temp:
                                        s10 = '--temp ' + str(j)
                                        if b == 625 and c == 64:
                                            continue
                                        if b == 676 and c == 55:
                                            continue

                                        job_name = [
                                            a, b, c, d, e, f, g, h, i, j
                                        ]

                                        job_name_str = '_'.join(
                                            [str(item) for item in job_name])

                                        final_s = [
                                            *s, s1, s2, s3, s4, s5, s6, s7, s8,
                                            s9, s10
                                        ]
                                        file_name = create_script(
                                            job_name_str, final_s)
                                        os.system(f'sbatch {file_name}')
                                        count = count + 1

print(f"Number of slurm scripts generated is: {count}")
