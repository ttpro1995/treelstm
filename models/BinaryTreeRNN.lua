--[[

  A Binary Tree-LSTM with input at the leaf nodes.

--]]

local BinaryTreeRNN, parent = torch.class('treelstm.BinaryTreeRNN', 'treelstm.TreeLSTM')

function BinaryTreeRNN:__init(config)
    parent.__init(self, config)
    --self.gate_output = config.gate_output
    --if self.gate_output == nil then self.gate_output = true end

    -- a function that instantiates an output module that takes the hidden state h as input
    self.output_module_fn = config.output_module_fn
    self.criterion = config.criterion

    -- leaf input module
    self.leaf_module = self:new_leaf_module()
    self.leaf_modules = {}

    -- composition module
    self.composer = self:new_composer()
    self.composers = {}

    -- output module
    self.output_module = self:new_output_module()
    self.output_modules = {}
end

function BinaryTreeRNN:new_leaf_module()
    local input = nn.Identity()()
    local c = nn.Linear(self.in_dim, self.mem_dim)(input)
    local h = nn.Tanh()(c)
    local leaf_module = nn.gModule({ input }, { h })
    if self.leaf_module ~= nil then
        share_params(leaf_module, self.leaf_module)
    end
    return leaf_module
end

function BinaryTreeRNN:new_composer()
    local lh = nn.Identity()()
    local rh = nn.Identity()()
    local new_gate = function()
        return nn.CAddTable() {
            nn.Linear(self.mem_dim, self.mem_dim)(lh),
            nn.Linear(self.mem_dim, self.mem_dim)(rh)
        }
    end
    local update = nn.Tanh()(new_gate()) -- memory cell update vector
    local composer = nn.gModule({lh, rh}, { update })

    if self.composer ~= nil then
        share_params(composer, self.composer)
    end
    return composer
end

function BinaryTreeRNN:new_output_module()
    if self.output_module_fn == nil then return nil end
    local output_module = self.output_module_fn()
    if self.output_module ~= nil then
        share_params(output_module, self.output_module)
    end
    return output_module
end

function BinaryTreeRNN:forward(tree, inputs) --recursive
    local lloss, rloss = 0, 0
    if tree.num_children == 0 then
        self:allocate_module(tree, 'leaf_module')
        tree.state = tree.leaf_module:forward(inputs[tree.leaf_idx])
    else
        self:allocate_module(tree, 'composer')

        -- get child hidden states
        local lvecs, lloss = self:forward(tree.children[1], inputs) --call recursively
        local rvecs, rloss = self:forward(tree.children[2], inputs)
        -- compute state and output
        tree.state = tree.composer:forward { lvecs, rvecs }
    end

    local loss
    if self.output_module ~= nil then
        self:allocate_module(tree, 'output_module')
        tree.output = tree.output_module:forward(tree.state) -- dua h vao
        if self.train then
            loss = self.criterion:forward(tree.output, tree.gold_label) + lloss + rloss
        end --cummulative loss through recusive call
    end

    return tree.state, loss
end

function BinaryTreeRNN:backward(tree, inputs, grad)
    local grad_inputs = torch.Tensor(inputs:size())
    self:_backward(tree, inputs, grad, grad_inputs)
    return grad_inputs
end

function BinaryTreeRNN:_backward(tree, inputs, grad, grad_inputs)
    --grad is grad at output of of the tree
    --output_grad from error born only at tree.output
    --grad_inputs is saved at each leaf so that can probagate to word embed
    local output_grad = self.mem_zeros
    if tree.output ~= nil and tree.gold_label ~= nil then
        output_grad = tree.output_module:backward(tree.state, self.criterion:backward(tree.output, tree.gold_label))
    end
    self:free_module(tree, 'output_module')

    if tree.num_children == 0 then
        grad_inputs[tree.leaf_idx] = tree.leaf_module:backward(inputs[tree.leaf_idx],
            grad + output_grad)
        self:free_module(tree, 'leaf_module')
    else
        local lh, rh = self:get_child_states(tree)
        local composer_grad = tree.composer:backward({ lh, rh }, grad + output_grad)
        self:free_module(tree, 'compose')

        -- backward propagate to children
        self:_backward(tree.children[1], inputs, composer_grad[1], grad_inputs)
        self:_backward(tree.children[2], inputs, composer_grad[2], grad_inputs)
    end
    tree.state = nil
    tree.output = nil
end

function BinaryTreeRNN:parameters()
    local params, grad_params = {}, {}
    local cp, cg = self.composer:parameters()
    tablex.insertvalues(params, cp)
    tablex.insertvalues(grad_params, cg)
    local lp, lg = self.leaf_module:parameters()
    tablex.insertvalues(params, lp)
    tablex.insertvalues(grad_params, lg)
    if self.output_module ~= nil then
        local op, og = self.output_module:parameters()
        tablex.insertvalues(params, op)
        tablex.insertvalues(grad_params, og)
    end
    return params, grad_params
end

--
-- helper functions
--
--[[
function BinaryTreeRNN:unpack_state(state)
    local h
    if state == nil then
        h = self.mem_zeros, self.mem_zeros
    else
        h = unpack(state)
    end
    return h
end
]]--

function BinaryTreeRNN:get_child_states(tree)
    local lh, rh
    if tree.children[1] ~= nil then
        lh = tree.children[1].state
    end

    if tree.children[2] ~= nil then
        rh = tree.children[2].state
    end
    return lh, rh
end

function BinaryTreeRNN:clean(tree)
    tree.state = nil
    -- tree.output = nil
    self:free_module(tree, 'leaf_module')
    self:free_module(tree, 'composer')
    self:free_module(tree, 'output_module')
    for i = 1, tree.num_children do
        self:clean(tree.children[i])
    end
end
