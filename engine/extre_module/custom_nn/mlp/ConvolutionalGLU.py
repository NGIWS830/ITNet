import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
  
import warnings 
warnings.filterwarnings('ignore') 
from calflops import calculate_flops   
 
import torch
import torch.nn as nn

from engine.extre_module.ultralytics_nn.conv import Conv

class ConvolutionalGLU(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features  
      
        hidden_features = int(2 * hidden_features / 3)  

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)

        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),    
            act_layer()  
        )   
    

        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
            
        self.drop = nn.Dropout(drop)    
     
        self.conv1x1 = Conv(in_features, out_features, 1) if in_features != out_features else nn.Identity()     
    
    def forward(self, x):       
        x_shortcut = self.conv1x1(x)

        x, v = self.fc1(x).chunk(2, dim=1) 
  
        x = self.dwconv(x) * v
  
        x = self.drop(x)
  
        x = self.fc2(x)

        x = self.drop(x) 

        return x_shortcut + x    

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
  
    module = ConvolutionalGLU(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)     

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)    
    print(RESET)     
