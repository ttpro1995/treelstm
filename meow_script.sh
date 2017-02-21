# Standard
# th relatedness/main.lua --model lstm --dim 150 | tee "logs/relatedness_lstm_dim150_1"
#th relatedness/main.lua --model lstm --dim 150 | tee "logs/relatedness_lstm_dim150_3"
#th relatedness/main.lua --model lstm --dim 150 | tee "logs/relatedness_lstm_dim150_4"
# th relatedness/main.lua --model lstm --dim 150 | tee "logs/relatedness_lstm_dim150_5"

# Bidirectional
#th relatedness/main.lua --model bilstm --dim 150 | tee "logs/relatedness_bilstm_dim150_1"
#th relatedness/main.lua --model bilstm --dim 150 | tee "logs/relatedness_bilstm_dim150_2"
#th relatedness/main.lua --model bilstm --dim 150 | tee "logs/relatedness_bilstm_dim150_3"
#th relatedness/main.lua --model bilstm --dim 150 | tee "logs/relatedness_bilstm_dim150_4"
#th relatedness/main.lua --model bilstm --dim 150 | tee "logs/relatedness_bilstm_dim150_5"

# 2 layers lstm
# th relatedness/main.lua --model lstm --layers 2 --dim 108 | tee "logs/relatedness_lstm_layers2_dim150_1"
# th relatedness/main.lua --model lstm --layers 2 --dim 108 | tee "logs/relatedness_lstm_layers2_dim150_2"
# th relatedness/main.lua --model lstm --layers 2 --dim 108 | tee "logs/relatedness_lstm_layers2_dim150_3"
# th relatedness/main.lua --model lstm --layers 2 --dim 108 | tee "logs/relatedness_lstm_layers2_dim150_4"
# th relatedness/main.lua --model lstm --layers 2 --dim 108 | tee "logs/relatedness_lstm_layers2_dim150_5"

# 2 layers bilstm
#date "+%H:%M:%S   %d/%m/%y"
#th relatedness/main.lua --model bilstm --layers 2 --dim 108 | tee "logs/relatedness_bilstm_layers2_dim150_1"
#th relatedness/main.lua --model bilstm --layers 2 --dim 108 | tee "logs/relatedness_bilstm_layers2_dim150_2"
#date "+%H:%M:%S   %d/%m/%y"

th sentiment/testLoadModel.lua -m constituency -b -e 1 | tee "logs/sentiment_con_dim150_binary_print_error_sent"

