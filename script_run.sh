export CUDA_VISIBLE_DEVICES=1

##### DQN #####
#python dqn.py --weights weights/breakout/good.pt
python dqn.py --game 'seaquest'

##### Average DQN #####
#python avg_dqn.py --game 'asterix'

##### Convert Tensorboard to CSV #####
#python tflogs2pandas.py experiments/only_logs/ --write-csv --no-write-pkl -o experiments/only_logs_csv/

##### Plot Curves #####
#python plot.py
