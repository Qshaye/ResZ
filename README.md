
## ResZ for MARL value Factorization
This code is build based on PyMARL2 and PyMARL. We assume that you have experience with PyMARL.
The requirements are the same as PyMARL2.

**In fact, this is an improved version of the code used in the paper, for communication purposes only!**

**The original code is available as open source at: https://github.com/xmu-rl-3dv/ResQ**


## Run an experiment 


```shell
cd ResZ
python3 main.py --config=ResZ --env-config=sc2 with env_args.map_name=2s3z
```


The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.


## Citing ResQ/ResZ 

If you use ResQ/ResZ in your research, please cite our paper

*Siqi Shen, Mengwei Qiu, Jun Liu, Weiquan Liu, Yongquan Fu, Xinwang Liu, Cheng Wang. ResQ: A Residual Q Function-based Approach for Multi-Agent Reinforcement Learning Value Factorization. NeurIPS 2022*

https://openreview.net/forum?id=bdnZ_1qHLCW

In BibTeX format:

```tex
@inproceedings{ResQ, 
author = {Siqi Shen and Mengwei Qiu and Jun Liu and Weiquan Liu and Yongquan Fu and Xinwang Liu and Cheng Wang}, 
title = {ResQ: A Residual Q Function-based Approach for Multi-Agent Reinforcement Learning Value Factorization}, 
booktitle = {{NeurIPS}}, 
year = {2022}}
```

## License

Code licensed under the Apache License v2.0
