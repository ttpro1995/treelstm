--
-- Created by IntelliJ IDEA.
-- User: vdvinh
-- Date: 10/24/16
-- Time: 11:37 PM
-- To change this template use File | Settings | File Templates.
--
require('..')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

-- read command line arguments
local args = lapp [[
Training script for sentiment classification on the SST dataset.
  -m,--model  (default constituency) Model architecture: [constituency, lstm, bilstm]
  -l,--layers (default 1)            Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)          LSTM memory dimension
  -e,--epochs (default 10)           Number of training epochs
  -b,--binary                        Train and evaluate on binary sub-task
]]

-- binary or fine-grained subtask
local fine_grained = not args.binary

-- directory containing dataset files
local data_dir = 'data/sst/'

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

print('loading datasets')
local test_dir = data_dir .. 'test/'
local dependency = (args.model == 'dependency')
local test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, fine_grained, dependency)
dataset = test_dataset

function nums_to_sentence(input_nums, vocab)
  --input_nums: dataset.sents[index]
  --vocab : dataset.vocab

  sent = ''
  for i = 1, input_nums:size(1) do
    sent = sent..' '..vocab:token(input_nums[i])
  end
  return sent
end
