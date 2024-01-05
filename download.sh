link=https://vnueduvn-my.sharepoint.com/:u:/g/personal/21020205_vnu_edu_vn/EQS80S1gZa5KuKxmTiWW1UMBE9RzEWuzTp2uVU9QV3O9hA?e=bEU5aB&download=1
filename=vocab.zip
cd data
if [ ! -d  "embedding" ]; then
    mkdir embedding
    cd embedding
    wget "$link" -O $filename
    unzip $filename

