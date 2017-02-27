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

function nums_to_words(input_nums, vocab)
  --input_nums: dataset.sents[index]
  --vocab : dataset.vocab

  words = {}
  for i = 1, input_nums:size(1) do
    table.insert(words, vocab:token(input_nums[i]))
  end
  return words
end

function nums_to_sentence(input_nums, vocab)
  --input_nums: dataset.sents[index]
  --vocab : dataset.vocab

  sent = ''
  for i = 1, input_nums:size(1) do
    sent = sent..' '..vocab:token(input_nums[i])
  end
  return sent
end

function getAllData(t, prevData)
  -- if prevData == nil, start empty, otherwise start with prevData
  local data = prevData or {}

  -- copy all the attributes from t
  for k,v in pairs(t) do
    data[k] = data[k] or v
  end

  -- get t's metatable, or exit if not existing
  local mt = getmetatable(t)
  if type(mt)~='table' then return data end

  -- get the __index from mt, or exit if not table
  local index = mt.__index
  if type(index)~='table' then return data end

  -- include the data from index into data, recursively, and return
  return getAllData(index, data)
end

function accuracy_print_error(pred, gold, nums_sent, vocab)
  correct_index = torch.eq(pred, gold)
  for i = 1, correct_index:size(1) do
    t = correct_index[i]
    if t == 0 then
      incorrect_num_sent = nums_sent[i]
      sent = tostring(i)
      pred_i = pred[i]
      gold_i = gold[i]
      sent = sent .. ' '.. pred_i .. ' '.. gold_i .. ' ' .. nums_to_sentence(incorrect_num_sent, vocab)
      print (sent)
    end
  end
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

args.model = "constituency"
args.binary = true
args.epochs = 1


local model_name, model_class, model_structure
model_name = 'Constituency Tree LSTM'
model_class = treelstm.TreeLSTMSentiment
local loaded = model_class.load("/media/vdvinh/25A1FEDE380BDADA/thien_code/treelstm/trained_models/sent-constituency.2class.1l.150d.1.th")
print("11111")
model_structure = args.model
header(model_name .. ' for Sentiment Classification')

-- binary or fine-grained subtask
local fine_grained = not args.binary

-- directory containing dataset files
local data_dir = 'data/sst/'

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

print('loading datasets')
local test_dir = data_dir .. 'test/'
local dev_dir = data_dir .. 'dev/'
local dependency = (args.model == 'dependency')
test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, fine_grained, dependency)


printf('num test  = %d\n', test_dataset.size)

-- evaluate
header('Evaluating on test set')
best_dev_model = loaded


---------------------------------------------------------------------USE THE CODE FROM HERE ___________________________________________
---------------------RUN meow_script.sh ----------------------------------------------------


sentence_index = 101

s_sent = test_dataset.sents[sentence_index]
s_tree = test_dataset.trees[sentence_index]
words = nums_to_words(s_sent, vocab)
prediction = best_dev_model:predict(s_tree, s_sent)

function print_tree_output(tree, level)
  -- print root first
  output = tree.output
  prediction = (output[1] > output[3]) and 1 or 3
  indent = ''
  for i = 0, level do
    indent = indent ..'  '
  end
  if tree ~= nil and tree.num_children == 0 then
    print(indent..prediction..' '..words[tree.leaf_idx])
  else
    print(indent..prediction)
  end

  -- recursively print child of that root
  for i = 1, #tree.children do
    print_tree_output(tree.children[i], level+1)
  end
end

print_tree_output(s_tree, 0)
sent = nums_to_sentence(s_sent, vocab)
print(sent)
