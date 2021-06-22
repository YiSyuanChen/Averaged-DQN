export CUDA_VISIBLE_DEVICES=1

##### DQN #####
#python dqn.py --weights weights/breakout/good.pt
#python dqn.py

##### Average DQN #####
python avg_dqn.py --game 'asterix'
