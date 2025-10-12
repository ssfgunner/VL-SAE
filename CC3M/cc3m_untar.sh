# untar the cc3m dataset

mkdir ./cc3m_data

for file_idx in {0..575}
do
    file_idx_name=$(printf "%04d" $file_idx)
    file_name=cc3m-train-${file_idx_name}.tar
    file_path=./cc3m-wds/${file_name}
    echo ${file_path}
    tar -xvf ${file_path}  -C ./cc3m_data
done