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

  - CIFAR-100 : https://drive.google.com/file/d/1nt-ZpkyJN-LQsHemQduLkenDErOPDcrF/view?usp=sharing
  - CIFAR-10  : https://drive.google.com/file/d/1GgQ7QjovoU4FWMr_c9NesgVdOakyGv9a/view?usp=sharing

- Unzip PTR & move file to directory

  - ex) CIFAR-10, PresB-Net-18

    ```shell
    tar -zxvf cifar_10.tar.gz
    cp -r ptr_file/propose_ptr_18 ./
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

  
