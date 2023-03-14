#!/bin/bash

if [ ! -d "./venv" ]; then
  python3 -m venv ./venv
fi

# detect the operating system
if [[ $(uname) == "Linux" ]]; then
  source venv/bin/activate
elif [[ $(uname) == "Darwin" ]]; then
  source venv/bin/activate
elif [[ $(uname) == "MINGW"* ]]; then
  source venv/Scripts/activate
elif [[ $(uname) == "MSYS"* ]]; then
  source venv/Scripts/activate
else
  echo "Unsupported operating system"
  exit 1
fi

pip install -r requirements.txt

# Train FFNN
python starter.py -model FFNN -batch_size 32 -window 10 -epochs 10 -savename FFNN_model -trainname mix.train.txt -validname mix.valid.txt -testname mix.test.txt  
# Train LSTM
python starter.py -model LSTM -batch_size 32 -window 5 -epochs 40 -savename LSTM_model -trainname mix.train.txt -validname mix.valid.txt -testname mix.test.txt
# classify ffnn 
python starter.py -model FFNN_CLASSIFY -batch_size 32 -window 10 -loadname FFNN_model 
# classify lstm
python starter.py -model LSTM_CLASSIFY -batch_size 32 -window 5 -loadname LSTM_model
# classify blind
python starter.py -model BLIND -batch_size 32 -window 10 -loadname ffnnModel