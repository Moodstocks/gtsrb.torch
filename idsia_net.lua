local networks = require 'networks'

local idsia_net = {}

function idsia_net.get_network(opt)
  local network = nn.Sequential()

  local nbr_elements = {}
  for c in string.gmatch(opt.cnn, "%d+") do
    nbr_elements[#nbr_elements + 1] = tonumber(c)
  end
  assert(#nbr_elements == 4,
      'opt.cnn should contain 4 comma separated values when working with '..
      'the idsia network, got '..#nbr_elements)


  local conv1 = networks.new_conv(3,nbr_elements[1], false, opt.no_cnorm, 7)
  local conv2 = networks.new_conv(nbr_elements[1],nbr_elements[2],
                                  false, opt.no_cnorm, 4)
  local conv3 = networks.new_conv(nbr_elements[2],nbr_elements[3],
                                  false, opt.no_cnorm, 4)

  local conv_output_size = networks.convs_noutput({conv1,conv2,conv3})

  local fc = networks.new_fc(conv_output_size, nbr_elements[4])
  local classifier = networks.new_classifier(nbr_elements[4],
                                             networks.nbr_classes)

  if opt.st and opt.locnet and opt.locnet ~= '' then
    network:add(networks.new_spatial_tranformer(opt.locnet,
                                                opt.rot, opt.sca, opt.tra,
                                                nil, nil,
                                                opt.no_cuda))
  end
  network:add(conv1)
  if opt.locnet2 and opt.locnet2 ~= '' then
    local _,current_size = networks.convs_noutput({conv1})
    network:add(networks.new_spatial_tranformer(opt.locnet2,
                                                opt.rot, opt.sca, opt.tra,
                                                current_size, nbr_elements[1],
                                                opt.no_cuda))
  end
  network:add(conv2)
  if opt.locnet3 and opt.locnet3 ~= '' then
    local _,current_size = networks.convs_noutput({conv1, conv2})
    network:add(networks.new_spatial_tranformer(opt.locnet3,
                                                opt.rot, opt.sca, opt.tra,
                                                current_size, nbr_elements[2],
                                                opt.no_cuda))
  end
  network:add(conv3)
  network:add(fc)
  network:add(classifier)

  return network
end

return idsia_net
