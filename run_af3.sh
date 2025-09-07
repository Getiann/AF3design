#!/bin/bash
#source "/venv/alphafold3_venv/bin/activate" #运行af3的环境 不用动
MODEL_PARAMS_DIR="/data/share/alphafold3" # af3参数和数据库所在的目录 不用动
INPUT_DIR="/home/ge/input" # input json(见下一条) 所在的路径 需要更改为自己的路径
OUTPUT_DIR="/home/ge/temp" #输出结果的路径 需要更改为自定义的输出结果存放路径
mkdir -p ${OUTPUT_DIR}


for dir in `find ${INPUT_DIR} -type d`; do

    if [[ "$dir" == "$INPUT_DIR" ]] && find "$dir" -mindepth 1 -type d | grep -q .; then
        echo "Skip the start directory... ..."
        continue
    elif [ -d "${dir}" ]; then
        echo "Current dir: ${dir}"

        current_output_dir=$OUTPUT_DIR
        mkdir -p ${current_output_dir}
        echo "Current output dir: ${current_output_dir}"

        for filename in `find ${dir} -type f`; do
            echo "Current data JSON file: ${filename}"
            python /home/ge/app/af3design/run_af3design_1.py \
                 --json_path=${filename} \
                --model_dir=${MODEL_PARAMS_DIR} \
                --output_dir=${current_output_dir} \ #single_sequence为false,MSA为True
                #--db_dir=${MODEL_PARAMS_DIR} 搜库指定的数据库所在的目录 搜库所必须的
        done
    fi
done
