require 'xlua'
local bench_utils = {}

-- Read the last line of a command line output handler
local read_last_line = function(handler)
  -- hack to get the last line of a file
  local last_line
  for line in handler:lines() do
    last_line = line
  end
  return last_line
end

-- Print one line to the output file
local print_one_result = function(output, command, accuracy)
  output:write("| "..command.." | "..accuracy.." |\n")
end

-- Run the tests specified in params and write the results in the output file
-- Each element of params must contain:
-- el[1] = command line option
-- el[2] = table of the values to test
-- if the value is false, the option will not be use,
-- if the value is true, the option will be used without specific value
function bench_utils.run_test(params, output)
  -- Add an empty line to the output file
  print_one_result(output,'','')

  -- Compute all the combinations
  local arg_table
  for _, param in pairs(params) do
    local new_table = {}
    for _, value in pairs(param[2]) do
      local param_value_opt = ''
      if value == true then
        param_value_opt = param[1]
      elseif value ~= false then
        param_value_opt = param[1] .. value
      end
      if not arg_table then
        table.insert(new_table, param_value_opt)
      else
        for _, arg in pairs(arg_table) do
          table.insert(new_table, arg .. ' ' .. param_value_opt)
        end
      end
    end
    arg_table = new_table
  end

  -- Run the tests
  local current = 0
  for _,arg in pairs(arg_table) do
    xlua.progress(current, #arg_table)
    current = current + 1

    local cmd = 'luajit main.lua --script ' .. arg
    local handler = io.popen(cmd)
    local accuracy = read_last_line(handler)

    print_one_result(output, arg, accuracy)
  end
  -- Finish progress
  xlua.progress(#arg_table, #arg_table)
end

return bench_utils
