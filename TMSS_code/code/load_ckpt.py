import torch

model = torch.load("./last_edit.ckpt")
list_keys = list(model['state_dict'].keys())
#list_keys = list(model.keys())
for i in list_keys:
    print(i)
#print(model['loops'].keys())
# del model['state_dict']['model.encoder2.blocks.0.1.conv3.conv.weight']
# del model['state_dict']['model.encoder2.blocks.1.1.conv3.conv.weight']
# del model['state_dict']['model.encoder3.blocks.0.1.conv3.conv.weight']
# torch.save(model, "./last_edit.ckpt")
