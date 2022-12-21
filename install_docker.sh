pip install -r requirements.txt
python -m spacy download en_core_web_lg
apt-get -y update
apt-get install git
apt-get install wget
apt-get install nano
apt-get install unzip
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
bash
