require 'nn'
require 'LSTM'

model = nn.Sequential()
model:add(treelstm.LSTM)

print(model)
