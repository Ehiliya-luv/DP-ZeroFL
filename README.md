# Zeroth-Order Federated Learning under Local Differential Privacy: Clipping Mechanisms and Convergence Analysis (DP-ZeroFL)

![ci](https://github.com/ZidongLiu/FedDisco/actions/workflows/ci.yaml/badge.svg) ![apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

DeComFL is a library designed for training/fine-tuning deep learning models in the federated learning scenario. Its unique feature is the utilization of zeroth-order optimization, enabling communication between clients to be limited to just a few scalars, irrespective of the original model's size. This dimension-free communication is the inspiration behind the library's name.

## Environment Setup
Open project and run code: `pip install -r requirements.txt`

## Run Experiments

- **Run zeroth-order random gradient estimate + SGD training**. Train model using ZOO RGE.
  Usage example: `python zo_rge_main.py --dataset=mnist --num-pert=10 --lr=1e-5 --mu=1e-3 --momentum=0.9`

- **Run DP-ZeroFL:** Follow FL routine, split data into chunks and train on different clients.
  Usage example: `python dpzerofl_main.py --large-model=opt-125m --dataset=sst2 --iterations=1000 --train-batch-size=32 --test-batch-size=200 --eval-iterations=25 --num-clients=3 --num-sample-clients=2 --local-update-steps=1 --num-pert=5 --lr=1e-5 --mu=1e-3 --grad-estimate-method=rge-forward` --clip_method=fixed

- **Run FedAvg:** Run standard fedavg algorithm.
  `python fo_fl_main.py --dataset=sst2 --lr=1e-3 --num-clients=5 --num-sample-clients=3 --local-update-steps=1 --train-batch-size=32 --test-batch-size=200 --momentum=0.9`




      <img src="https://github.com/user-attachments/assets/c0dfb199-0a51-4b17-b9ba-9fe09d2c4f7a" alt="Image 2" style="width: 51%;" /> &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="https://github.com/user-attachments/assets/23ba00dc-fc62-4ab3-9c70-0326aa20b786" alt="Image 3" style="width: 25%;" />
    </div>
</div>
