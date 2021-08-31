# PresB-net







## Requirements

| Library | version |
| ------- | ------- |
| pytorch | 1.7.1   |
| numpy   | 1.19.2  |
| tqdm    | 4.59.0  |
| Pillow  | 8.1.2   |

## How to run

#### Use Pretrain weight

- Download weight(ptr) files

  - CIFAR-10  : https://drive.google.com/file/d/1ges05vLF-P3vw1vQ3PkPLY3mmOuDaDUa/view?usp=sharing
  - CIFAR-100 : https://drive.google.com/file/d/1Y5fFRQapvVLRdViHKPXdVm6u7bPQPOoh/view?usp=sharing

- Unzip PTR & move file to directory

  - CIFAR-10

    ```shell
    cd CIFAR10/PreB-net/
    tar -zxvf cifar_10.tar.gz
    ```

  - CIFAR-100

    ```shell
    cd CIFAR100/PreB-net/
    tar -zxvf cifar_100.tar.gz	
    ```



- Change model sturcture

  Change stage_channel parameter in Class proposed in models.py

  ```python
  PresB_10_channel = (64,64,128,256,512)
  PresB_18_channel = (64,64,64,128,128,256,256,512,512)
  PresB_34_channel = (64,64,64,64,128,128,128,128,256,256,256,256,256,256,512,512,512)
  
  class proposed(nn.Module):    
      def __init__(self, stage_channel = PresB_18_channel ,binarized = False ,num_class = 10):
          
          stage_channel = [128] + [2*i for i in stage_channel[1:]]
  ```

  

- Run

  ```shell
  sh custom_train_shell.sh
  ```

  

#### Training

- Make directory

  ```shell
  mkdir ${FILENAME}
  mkdir ${FILENAME}/version_1
  ```

  

- Edit shell

  - custon_train_shell.sh

  ```shell
  SAVEDIR="./${FILENAME}/version_1"
  DATAPATH="${DATAPATH}"
  ```

- Run

  ```shell
  sh custom_train_shell.sh
  ```

  
